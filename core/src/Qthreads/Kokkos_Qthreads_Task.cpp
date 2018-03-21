/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact  H. Carter Edwards (hcedwar@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <Kokkos_Macros.hpp>
#if defined( KOKKOS_ENABLE_QTHREADS ) && defined( KOKKOS_ENABLE_TASKPOLICY )

#include <Kokkos_Core.hpp>
#include <impl/Kokkos_TaskQueue_impl.hpp>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
  namespace Impl {

    template class TaskQueue< Kokkos::Qthreads > ;

    class HostThreadTeamDataSingleton : private HostThreadTeamData {
    private:

      HostThreadTeamDataSingleton() : HostThreadTeamData()
      {
        Kokkos::Qthreads::memory_space space ;
        const size_t num_pool_reduce_bytes  =   32 ;
        const size_t num_team_reduce_bytes  =   32 ;
        const size_t num_team_shared_bytes  = 1024 ;
        const size_t num_thread_local_bytes = 1024 ;
        const size_t alloc_bytes =
          HostThreadTeamData::scratch_size( num_pool_reduce_bytes
                                            , num_team_reduce_bytes
                                            , num_team_shared_bytes
                                            , num_thread_local_bytes );

        HostThreadTeamData::scratch_assign
          ( space.allocate( alloc_bytes )
            , alloc_bytes
            , num_pool_reduce_bytes
            , num_team_reduce_bytes
            , num_team_shared_bytes
            , num_thread_local_bytes );
      }

      ~HostThreadTeamDataSingleton()
      {
        Kokkos::Qthreads::memory_space space ;
        space.deallocate( HostThreadTeamData::scratch_buffer()
                          , HostThreadTeamData::scratch_bytes() );
      }

    public:

      static HostThreadTeamData & singleton()
      {
        static HostThreadTeamDataSingleton s ;
        return s ;
      }
    };


    //----------------------------------------------------------------------------

    TaskExec< Kokkos::Qthreads >::TaskExec()
      : m_self_exec( 0 ),
        m_team_exec( 0 ),
        m_sync_mask( 0 ),
        m_sync_value( 0 ),
        m_sync_step( 0 ),
        m_group_rank( 0 ),
        m_team_rank( 0 ),
        m_team_size( 1 )
    {}

    TaskExec< Kokkos::Qthreads >::
    TaskExec( Kokkos::Impl::QthreadsExec & arg_exec, int const arg_team_size )
      : m_self_exec( & arg_exec ),
        m_team_exec( arg_exec.pool_rev(arg_exec.pool_rank_rev() / arg_team_size) ),
        m_sync_mask( 0 ),
        m_sync_value( 0 ),
        m_sync_step( 0 ),
        m_group_rank( arg_exec.pool_rank_rev() / arg_team_size ),
        m_team_rank( arg_exec.pool_rank_rev() % arg_team_size ),
        m_team_size( arg_team_size )
    {
      // This team spans
      //    m_self_exec->pool_rev( team_size * group_rank )
      //    m_self_exec->pool_rev( team_size * ( group_rank + 1 ) - 1 )

      int64_t volatile * const sync = (int64_t *) m_self_exec->scratch_reduce();

      sync[0] = int64_t(0) ;
      sync[1] = int64_t(0) ;

      for ( int i = 0 ; i < m_team_size ; ++i ) {
        m_sync_value |= int64_t(1) << (8*i);
        m_sync_mask  |= int64_t(3) << (8*i);
      }

      Kokkos::memory_fence();
    }

#if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )

    void TaskExec< Kokkos::Qthreads >::team_barrier() const
    {
      if ( 1 < m_team_size ) {

        if ( m_team_exec->scratch_reduce_size() < int(2 * sizeof(int64_t)) ) {
          Kokkos::abort("TaskQueue<Qthreads> scratch_reduce memory too small");
        }

        // Use team shared memory to synchronize.
        // Alternate memory locations between barriers to avoid a sequence
        // of barriers overtaking one another.

        int64_t volatile * const sync =
          ((int64_t *) m_team_exec->scratch_reduce()) + ( m_sync_step & 0x01 );

        // This team member sets one byte within the sync variable
        int8_t volatile * const sync_self =
          ((int8_t *) sync) + m_team_rank ;

#if 0
        fprintf( stdout,
                 "barrier group(%d) member(%d) step(%d) wait(%lx) : before(%lx)\n",
                 m_group_rank,
                 m_team_rank,
                 m_sync_step,
                 m_sync_value,
                 *sync
                 );
        fflush(stdout);
#endif

        *sync_self = int8_t( m_sync_value & 0x03 ); // signal arrival

        while ( m_sync_value != *sync ); // wait for team to arrive

#if 0
        fprintf( stdout,
                 "barrier group(%d) member(%d) step(%d) wait(%lx) : after(%lx)\n",
                 m_group_rank,
                 m_team_rank,
                 m_sync_step,
                 m_sync_value,
                 *sync
                 );
        fflush(stdout);
#endif

        ++m_sync_step ;

        if ( 0 == ( 0x01 & m_sync_step ) ) { // Every other step
          m_sync_value ^= m_sync_mask ;
          if ( 1000 < m_sync_step ) m_sync_step = 0 ;
        }
      }
    }

#endif

    //----------------------------------------------------------------------------

    void TaskQueueSpecialization< Kokkos::Qthreads >::execute
    ( TaskQueue< Kokkos::Qthreads > * const queue )
    {
      using execution_space = Kokkos::Qthreads ;
      using queue_type      = TaskQueue< execution_space > ;
      using task_root_type  = TaskBase< execution_space, void, void > ;
      //using PoolExec        = Kokkos::Impl::QthreadsExec ;
      using Member          = TaskExec< execution_space > ;

      task_root_type * const end = (task_root_type *) task_root_type::EndTag ;

      // Required:  team_size <= 8

      HostThreadTeamData & team_data_single =
        HostThreadTeamDataSingleton::singleton();

      Impl::OpenQthreadsExec * instance = t_qthreads_instance;
      const int pool_size = Qthreads::thread_pool_size();

      //const int team_size = PoolExec::pool_size(2); // Threads per core
      // const int team_size = PoolExec::pool_size(1); // Threads per NUMA

      const int team_size = 1;  // Threads per core
      instance->resize_thread_data( 0 /* global reduce buffer */
                                    , 512 * team_size /* team reduce buffer */
                                    , 0 /* team shared buffer */
                                    , 0 /* thread local buffer */
                                    );

      /*
        if ( 8 < team_size ) {
        Kokkos::abort("TaskQueue<Qthreads> unsupported team size");
        }
      */
      //#pragma omp parallel
        {
          //PoolExec & self = *PoolExec::get_thread_omp();
          Impl::HostThreadTeamData & self = *(instance->get_thread_data());

          if ( self.organize_team( team_size ) ) {

            Member single_exec;
            Member team_exec( self, team_size );

            task_root_type * task = 0 ;

            do {
              // Each team lead attempts to acquire either a thread team task
              // or collection of single thread tasks for the team.

              if ( 0 == team_exec.team_rank() ) {

                bool leader_loop = false ;

                do {
                  if ( 0 != task && end != task ) {
                    queue->complete( task );
                  }
                }
                task = 0 < *((volatile int *) & queue->m_ready_count) ? end : 0 ;

                // Loop by priority and then type
                for ( int i = 0 ; i < queue_type::NumQueue && end == task ; ++i ) {
                  for ( int j = 0 ; j < 2 && end == task ; ++j ) {
                    task = queue_type::pop_read_task( & queue->m_ready[i][j] );
                  }
                }

                // If still tasks are still executing
                // and no task could be acquired
                // then continue this leader loop
                leader_loop = end == task ;

                if ( ( ! leader_loop ) &&
                     ( 0 != task ) &&
                     ( task_root_type::TaskSingle == task->m_task_type ) ) {

                  // if a single thread task then execute now
                  (*task->m_apply)( task , & single_exec );
                  leader_loop = true ;
                }
              } while ( leader_loop );
            }
            // Team lead either found 0 == m_ready_count or a team task
            // Team lead broadcast acquired task:

            team_exec.team_broadcast( task , 0);

            if ( 0 != task ) { // Thread Team Task

              (*task->m_apply)( task , & team_exec );

              // The m_apply function performs a barrier
            }
          } while( 0 != task );
        }
        self.disband_team();
    }
  }

  void TaskQueueSpecialization< Kokkos::Qthreads >::
  iff_single_thread_recursive_execute
  ( TaskQueue< Kokkos::Qthreads > * const queue )
  {
    using execution_space = Kokkos::Qthreads ;
    using queue_type      = TaskQueue< execution_space > ;
    using task_root_type  = TaskBase< execution_space, void, void > ;
    using Member          = TaskExec< execution_space > ;

    if ( 1 == Qthreads::thread_pool_size() ) {

      task_root_type * const end = (task_root_type *) task_root_type::EndTag ;

      HostThreadTeamData & team_data_single =
        HostThreadTeamDataSingleton::singleton();

      Member single_exec( team_data_single );

      task_root_type * task = end ;

      do {

        task = end ;

        // Loop by priority and then type
        for ( int i = 0 ; i < queue_type::NumQueue && end == task ; ++i ) {
          for ( int j = 0 ; j < 2 && end == task ; ++j ) {
            task = queue_type::pop_task( & queue->m_ready[i][j] );
          }
        }

        if ( end == task ) break ;

        (*task->m_apply)( task, & single_exec );

        queue->complete( task );

      } while(1);
    }
  }

}} /* namespace Kokkos::Impl */

//----------------------------------------------------------------------------
#else
void KOKKOS_SRC_QTHREADS_TASK_PREVENT_LINK_ERROR() {}
#endif /* #if defined( KOKKOS_ENABLE_QTHREADS ) && defined( KOKKOS_ENABLE_TASKPOLICY ) */
