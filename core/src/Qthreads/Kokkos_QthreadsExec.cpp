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
#if defined( KOKKOS_ENABLE_QTHREADS )

#include <Kokkos_Core_fwd.hpp>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <utility>

#include <Kokkos_Core.hpp>

#include <Kokkos_Qthreads.hpp>
#include <Kokkos_Atomic.hpp>
#include <impl/Kokkos_Error.hpp>

// Defines to enable experimental Qthreads functionality.
#define QTHREAD_LOCAL_PRIORITY
#define CLONED_TASKS

#include <qthread/qthread.h>

//----------------------------------------------------------------------------

namespace Kokkos {

  enum { MAXIMUM_QTHREADS_WORKERS = 1024 };
  namespace Impl {

int g_qthreads_hardware_max_threads = 1;

__thread int t_qthreads_hardware_id = 0;
__thread Impl::QthreadsExec * t_qthreads_instance = nullptr;


/** s_exec is indexed by the reverse rank of the workers
 *  for faster fan-in / fan-out lookups
 *  [ n - 1, n - 2, ..., 0 ]
 */
QthreadsExec * s_exec[ MAXIMUM_QTHREADS_WORKERS ];

int  s_number_shepherds            = 0;
int  s_number_workers_per_shepherd = 0;
int  s_number_workers              = 0;

inline
QthreadsExec ** worker_exec()
{
  return s_exec + s_number_workers - ( qthread_shep() * s_number_workers_per_shepherd + qthread_worker_local( NULL ) + 1 );
}

const int s_base_size = QthreadsExec::align_alloc( sizeof(QthreadsExec) );

int s_worker_reduce_end   = 0;  // End of worker reduction memory.
int s_worker_shared_end   = 0;  // Total of worker scratch memory.
int s_worker_shared_begin = 0;  // Beginning of worker shared memory.

QthreadsExecFunctionPointer volatile s_active_function     = 0;
const void                * volatile s_active_function_arg = 0;

void QthreadsExec::validate_partition( const int nthreads
                                   , int & num_partitions
                                   , int & partition_size
                                  )
{
  if (nthreads == 1) {
    num_partitions = 1;
    partition_size = 1;
  }
  else if( num_partitions < 1 && partition_size < 1) {
    int idle = nthreads;
    for (int np = 2; np <= nthreads ; ++np) {
      for (int ps = 1; ps <= nthreads/np; ++ps) {
        if (nthreads - np*ps < idle) {
          idle = nthreads - np*ps;
          num_partitions = np;
          partition_size = ps;
        }
        if (idle == 0) {
          break;
        }
      }
    }
  }
  else if( num_partitions < 1 && partition_size > 0 ) {
    if ( partition_size <= nthreads ) {
      num_partitions = nthreads / partition_size;
    }
    else {
      num_partitions = 1;
      partition_size = nthreads;
    }
  }
  else if( num_partitions > 0 && partition_size < 1 ) {
    if ( num_partitions <= nthreads ) {
      partition_size = nthreads / num_partitions;
    }
    else {
      num_partitions = nthreads;
      partition_size = 1;
    }
  }
  else if ( num_partitions * partition_size > nthreads ) {
    int idle = nthreads;
    const int NP = num_partitions;
    const int PS = partition_size;
    for (int np = NP; np > 0; --np) {
      for (int ps = PS; ps > 0; --ps) {
        if (  (np*ps <= nthreads)
           && (nthreads - np*ps < idle) ) {
          idle = nthreads - np*ps;
          num_partitions = np;
          partition_size = ps;
        }
        if (idle == 0) {
          break;
        }
      }
    }
  }

}

void QthreadsExec::verify_is_master( const char * const label )
{
  if ( !t_qthreads_instance )
  {
    std::string msg( label );
    msg.append( " ERROR: in parallel or not initialized" );
    Kokkos::Impl::throw_runtime_exception( msg );
  }
}


} // namespace Impl
} // namespace Kokkos


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
namespace Kokkos {
  namespace Impl {

    void QthreadsExec::clear_thread_data()
    {
      const size_t member_bytes =
        sizeof(int64_t) *
        HostThreadTeamData::align_to_int64( sizeof(HostThreadTeamData) );

      const int old_alloc_bytes =
        m_pool[0] ? ( member_bytes + m_pool[0]->scratch_bytes() ) : 0 ;

      Qthreads::memory_space space ;

      //#pragma omp parallel num_threads( m_pool_size )
      {
        const int rank = 1; //omp_get_thread_num();

        if ( 0 != m_pool[rank] ) {

          m_pool[rank]->disband_pool();

          space.deallocate( m_pool[rank] , old_alloc_bytes );

          m_pool[rank] = 0 ;
        }
      }
      /* END #pragma omp parallel */
    }

void QthreadsExec::resize_thread_data( size_t pool_reduce_bytes
                                   , size_t team_reduce_bytes
                                   , size_t team_shared_bytes
                                   , size_t thread_local_bytes )
{
  const size_t member_bytes =
    sizeof(int64_t) *
    HostThreadTeamData::align_to_int64( sizeof(HostThreadTeamData) );

  std::cout << "m_pool[0] " << m_pool[0] << std::endl;
#if 0
  //if(m_pool[0] == 0x3e8){
    m_pool[0] = nullptr;
    // Qthreads::memory_space space0 ;
    // void * const ptr0 = space0.allocate( member_bytes );

    // m_pool[ 0 ] = new( ptr0 ) HostThreadTeamData();

  }
#endif
  HostThreadTeamData * root = m_pool[0] ;

  const size_t old_pool_reduce  = root ? root->pool_reduce_bytes() : 0 ;
  const size_t old_team_reduce  = root ? root->team_reduce_bytes() : 0 ;
  const size_t old_team_shared  = root ? root->team_shared_bytes() : 0 ;
  const size_t old_thread_local = root ? root->thread_local_bytes() : 0 ;
  const size_t old_alloc_bytes  = root ? ( member_bytes + root->scratch_bytes() ) : 0 ;

  // Allocate if any of the old allocation is tool small:
  std::cout << "checking allocate " << std::endl;
  const bool allocate = ( old_pool_reduce  < pool_reduce_bytes ) ||
                        ( old_team_reduce  < team_reduce_bytes ) ||
                        ( old_team_shared  < team_shared_bytes ) ||
                        ( old_thread_local < thread_local_bytes );

  if ( allocate ) {

    if ( pool_reduce_bytes < old_pool_reduce ) { pool_reduce_bytes = old_pool_reduce ; }
    if ( team_reduce_bytes < old_team_reduce ) { team_reduce_bytes = old_team_reduce ; }
    if ( team_shared_bytes < old_team_shared ) { team_shared_bytes = old_team_shared ; }
    if ( thread_local_bytes < old_thread_local ) { thread_local_bytes = old_thread_local ; }

    std::cout << "scratch sizing" << std::endl;
    const size_t alloc_bytes =
      member_bytes +
      HostThreadTeamData::scratch_size( pool_reduce_bytes
                                      , team_reduce_bytes
                                      , team_shared_bytes
                                      , thread_local_bytes );

    Qthreads::memory_space space ;

    memory_fence();

    // do in parallel get affinity.
    //#pragma omp parallel num_threads(m_pool_size)
    for(int i = 0; i < m_pool_size; i++ ){
      const int rank =  i; //omp_get_thread_num();
      std::cout << "adding rank " << rank << std::endl;
      if ( 0 != m_pool[rank] ) {

        m_pool[rank]->disband_pool();

        space.deallocate( m_pool[rank] , old_alloc_bytes );
      }

      void * const ptr = space.allocate( alloc_bytes );

      m_pool[ rank ] = new( ptr ) HostThreadTeamData();

      m_pool[ rank ]->
        scratch_assign( ((char *)ptr) + member_bytes
                      , alloc_bytes
                      , pool_reduce_bytes
                      , team_reduce_bytes
                      , team_shared_bytes
                      , thread_local_bytes
                      );

      memory_fence();
    }
/* END #pragma omp parallel */

    HostThreadTeamData::organize_pool( m_pool , m_pool_size );
  }
}
} // namespace Impl
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------


namespace Kokkos {
//----------------------------------------------------------------------------

  int Qthreads::get_current_max_threads() noexcept
  {
    return MAXIMUM_QTHREADS_WORKERS;
  }

void Qthreads::initialize( int thread_count )
{
  // Environment variable: QTHREAD_NUM_SHEPHERDS
  // Environment variable: QTHREAD_NUM_WORKERS_PER_SHEP
  // Environment variable: QTHREAD_HWPAR
  if ( Impl::t_qthreads_instance )
    {
      finalize();
    }

  Qthreads::memory_space space ;

  std::cout << "INITIALIZING!!!" << std::endl;
  {
    char buffer[256];
    printf("thread_count %d\n", thread_count);
    snprintf( buffer, sizeof(buffer), "QTHREAD_HWPAR=%d", thread_count );
    putenv( buffer );
  }

  /* so what am I supposed to do with hwloc here? */

  printf("qthread_initialize %d\n", qthread_initialize() );
  printf("qthread_num_shepards %d qthread_num_workers_local( NO_SHEPHERD ) %d\n", qthread_num_shepherds(),   qthread_num_workers_local( NO_SHEPHERD ));
  printf("qthread_num_workers() %d\n", qthread_num_workers() );
  printf("thread_count %d\n", thread_count);
  // hack fixme.
  if(thread_count == -1)
    thread_count = 4;
  const bool ok_init = ( QTHREAD_SUCCESS == qthread_initialize() ) &&
                       ( thread_count    == qthread_num_shepherds() * qthread_num_workers_local( NO_SHEPHERD ) ) &&
                       ( thread_count    == qthread_num_workers() );

  bool ok_symmetry = true;
  printf("ok_init %d\n", ok_init);
  printf("ok_symmetry %d\n", ok_symmetry);


  if ( ok_init ) {
    Impl::s_number_shepherds            = qthread_num_shepherds();
    Impl::s_number_workers_per_shepherd = qthread_num_workers_local( NO_SHEPHERD );
    Impl::s_number_workers              = Impl::s_number_shepherds * Impl::s_number_workers_per_shepherd;

    for ( int i = 0; ok_symmetry && i < Impl::s_number_shepherds; ++i ) {
      ok_symmetry = ( Impl::s_number_workers_per_shepherd == qthread_num_workers_local( i ) );
    }
  }
  printf("ok_symmetry after %d\n", ok_symmetry);
  if ( ! ok_init || ! ok_symmetry ) {
    std::ostringstream msg;

    msg << "Kokkos::Qthreads::initialize(" << thread_count << ") FAILED";
    msg << " : qthread_num_shepherds = " << qthread_num_shepherds();
    msg << " : qthread_num_workers_per_shepherd = " << qthread_num_workers_local( NO_SHEPHERD );
    msg << " : qthread_num_workers = " << qthread_num_workers();

    if ( ! ok_symmetry ) {
      msg << " : qthread_num_workers_local = {";
      for ( int i = 0; i < Impl::s_number_shepherds; ++i ) {
        msg << " " << qthread_num_workers_local( i );
      }
      msg << " }";
    }

    Impl::s_number_workers              = 0;
    Impl::s_number_shepherds            = 0;
    Impl::s_number_workers_per_shepherd = 0;

    if ( ok_init ) { qthread_finalize(); }

    Kokkos::Impl::throw_runtime_exception( msg.str() );
  }

  Impl::g_qthreads_hardware_max_threads = qthread_num_shepherds() * qthread_num_workers(); // 512; //get_current_max_threads();

  void * const ptr = space.allocate( sizeof(Impl::QthreadsExec ));
  Impl::t_qthreads_instance = new (ptr) Impl::QthreadsExec( Impl::g_qthreads_hardware_max_threads );

  std::cout << "resizing thread data " << std::endl;
  // New, unified host thread team data:
  {
    size_t pool_reduce_bytes  =   32 * thread_count ;
    size_t team_reduce_bytes  =   32 * thread_count ;
    size_t team_shared_bytes  = 1024 * thread_count ;
    size_t thread_local_bytes = 1024 ;

    Impl::t_qthreads_instance->resize_thread_data( pool_reduce_bytes
                                                 , team_reduce_bytes
                                                 , team_shared_bytes
                                                 , thread_local_bytes
                                                 );
  }
  Impl::QthreadsExec::resize_worker_scratch( 1024, 1024 );


  //Impl::QthreadsExec::resize_worker_scratch( 256, 256 );

  // // Check for over-subscription
  // if( Impl::mpi_ranks_per_node() * long(thread_count) > Impl::processors_per_node() ) {
  //   std::cout << "Kokkos::Threads::initialize WARNING: You are likely oversubscribing your CPU cores." << std::endl;
  //   std::cout << "                                    Detected: " << Impl::processors_per_node() << " cores per node." << std::endl;
  //   std::cout << "                                    Detected: " << Impl::mpi_ranks_per_node() << " MPI_ranks per node." << std::endl;
  //   std::cout << "                                    Requested: " << thread_count << " threads per process." << std::endl;
  // }

  // Init the array for used for arbitrarily sized atomics.
  Impl::init_lock_array_host_space();

  Impl::SharedAllocationRecord< void, void >::tracking_enable();

#if defined(KOKKOS_ENABLE_PROFILING)
  Kokkos::Profiling::initialize();
#endif
}

void Qthreads::finalize()
{
  Impl::QthreadsExec::clear_workers();

  if ( Impl::s_number_workers ) {
    qthread_finalize();
  }

  Impl::s_number_workers              = 0;
  Impl::s_number_shepherds            = 0;
  Impl::s_number_workers_per_shepherd = 0;
}
  int Qthreads::is_initialized()
  {
    /*
      if(Impl::s_number_workers == 0)
      Qthreads::initialize(10);
    */
    return Impl::s_number_workers != 0;
  }

  int Qthreads::concurrency()
  {
    return Impl::s_number_workers_per_shepherd;
  }

  bool Qthreads::in_parallel()
  {
    return Impl::s_active_function != 0;
  }


void Qthreads::print_configuration( std::ostream & s, const bool detail )
{
  s << "Kokkos::Qthreads {"
    << " num_shepherds(" << Impl::s_number_shepherds << ")"
    << " num_workers_per_shepherd(" << Impl::s_number_workers_per_shepherd << ")"
    << " }" << std::endl;
}

Qthreads & Qthreads::instance( int )
{
  static Qthreads q;
  return q;
}

void Qthreads::fence()
{
}

int Qthreads::shepherd_size() const { return Impl::s_number_shepherds; }
int Qthreads::shepherd_worker_size() const { return Impl::s_number_workers_per_shepherd; }

const char* Qthreads::name() { return "Qthreads"; }

} // namespace Kokkos

//----------------------------------------------------------------------------

namespace Kokkos {

namespace Impl {

namespace {

aligned_t driver_exec_all( void * arg )
{
  QthreadsExec & exec = **worker_exec();

  (*s_active_function)( exec, s_active_function_arg );

/*
  fprintf( stdout
         , "QthreadsExec driver worker(%d:%d) shepherd(%d:%d) shepherd_worker(%d:%d) done\n"
         , exec.worker_rank()
         , exec.worker_size()
         , exec.shepherd_rank()
         , exec.shepherd_size()
         , exec.shepherd_worker_rank()
         , exec.shepherd_worker_size()
         );
  fflush(stdout);
*/

  return 0;
}

aligned_t driver_resize_worker_scratch( void * arg )
{
  static volatile int lock_begin = 0;
  static volatile int lock_end   = 0;

  QthreadsExec ** const exec = worker_exec();

  //----------------------------------------
  // Serialize allocation for thread safety.

  while ( ! atomic_compare_exchange_strong( & lock_begin, 0, 1 ) ); // Spin wait to claim lock.

  const bool ok = 0 == *exec;

  if ( ok ) { *exec = (QthreadsExec *) malloc( s_base_size + s_worker_shared_end ); }

  lock_begin = 0; // Release lock.

  if ( ok ) { new( *exec ) QthreadsExec( s_number_workers ); }

  //----------------------------------------
  // Wait for all calls to complete to insure that each worker has executed.

  if ( s_number_workers == 1 + atomic_fetch_add( & lock_end, 1 ) ) { lock_end = 0; }

  while ( lock_end );

/*
  fprintf( stdout
         , "QthreadsExec resize worker(%d:%d) shepherd(%d:%d) shepherd_worker(%d:%d) done\n"
         , (**exec).worker_rank()
         , (**exec).worker_size()
         , (**exec).shepherd_rank()
         , (**exec).shepherd_size()
         , (**exec).shepherd_worker_rank()
         , (**exec).shepherd_worker_size()
         );
  fflush(stdout);
*/

  //----------------------------------------

  if ( ! ok ) {
    fprintf( stderr, "Kokkos::QthreadsExec resize failed\n" );
    fflush( stderr );
  }

  return 0;
}

void verify_is_process( const char * const label, bool not_active = false )
{
  const bool not_process = 0 != qthread_shep() || 0 != qthread_worker_local( NULL );
  const bool is_active   = not_active && ( s_active_function || s_active_function_arg );

  if ( not_process || is_active ) {
    std::string msg( label );
    msg.append( " : FAILED" );
    if ( not_process ) msg.append(" : not called by main process");
    if ( is_active )   msg.append(" : parallel execution in progress");
    Kokkos::Impl::throw_runtime_exception( msg );
  }
}

} // namespace

int QthreadsExec::worker_per_shepherd()
{
  return s_number_workers_per_shepherd;
}

QthreadsExec::QthreadsExec()
{
  const int shepherd_rank        = qthread_shep();
  const int shepherd_worker_rank = qthread_worker_local( NULL );
  const int worker_rank          = shepherd_rank * s_number_workers_per_shepherd + shepherd_worker_rank;

  m_worker_base          = s_exec;
  m_shepherd_base        = s_exec + s_number_workers_per_shepherd * ( ( s_number_shepherds - ( shepherd_rank + 1 ) ) );
  m_scratch_alloc        = ( (unsigned char *) this ) + s_base_size;
  m_reduce_end           = s_worker_reduce_end;
  m_shepherd_rank        = shepherd_rank;
  m_shepherd_size        = s_number_shepherds;
  m_shepherd_worker_rank = shepherd_worker_rank;
  m_shepherd_worker_size = s_number_workers_per_shepherd;
  m_worker_rank          = worker_rank;
  m_worker_size          = s_number_workers;
  m_worker_state         = QthreadsExec::Active;
  Qthreads::memory_space space ;

  printf("allocating m_pool\n");
  void * const ptr = space.allocate( sizeof(Impl::QthreadsExec) );
  for(int i = 0; i < MAX_THREAD_COUNT; i++)
      m_pool[ i ] = new( ptr ) HostThreadTeamData();
  printf("allocated m_pool %p\n", &m_pool);
}

void QthreadsExec::clear_workers()
{
  for ( int iwork = 0; iwork < s_number_workers; ++iwork ) {
    QthreadsExec * const exec = s_exec[iwork];
    s_exec[iwork] = 0;
    free( exec );
  }
}

void QthreadsExec::shared_reset( Qthreads::scratch_memory_space & space )
{
  new( & space )
    Qthreads::scratch_memory_space(
      ((unsigned char *) (**m_shepherd_base).m_scratch_alloc ) + s_worker_shared_begin,
      s_worker_shared_end - s_worker_shared_begin
    );
}

void QthreadsExec::resize_worker_scratch( const int reduce_size, const int shared_size )
{
  const int exec_all_reduce_alloc = align_alloc( reduce_size );
  const int shepherd_scan_alloc   = align_alloc( 8 );
  const int shepherd_shared_end   = exec_all_reduce_alloc + shepherd_scan_alloc + align_alloc( shared_size );

  if ( s_worker_reduce_end < exec_all_reduce_alloc ||
       s_worker_shared_end < shepherd_shared_end ) {

/*
  fprintf( stdout, "QthreadsExec::resize\n");
  fflush(stdout);
*/

    // Clear current worker memory before allocating new worker memory.
    clear_workers();

    // Increase the buffers to an aligned allocation.
    s_worker_reduce_end   = exec_all_reduce_alloc;
    s_worker_shared_begin = exec_all_reduce_alloc + shepherd_scan_alloc;
    s_worker_shared_end   = shepherd_shared_end;

    // Need to query which shepherd this main 'process' is running.

    const int main_shep = qthread_shep();

    // Have each worker resize its memory for proper first-touch.
#if 0
    for ( int jshep = 0; jshep < s_number_shepherds; ++jshep ) {
      for ( int i = jshep != main_shep ? 0 : 1; i < s_number_workers_per_shepherd; ++i ) {
        qthread_fork_to( driver_resize_worker_scratch, NULL, NULL, jshep );
      }
    }
#else
    // If this function is used before the 'qthreads.task_policy' unit test,
    // the 'qthreads.task_policy' unit test fails with a seg-fault within libqthread.so.
    for ( int jshep = 0; jshep < s_number_shepherds; ++jshep ) {
      const int num_clone = jshep != main_shep ? s_number_workers_per_shepherd : s_number_workers_per_shepherd - 1;

      if ( num_clone ) {
        const int ret = qthread_fork_clones_to_local_priority
          ( driver_resize_worker_scratch   // Function
          , NULL                           // Function data block
          , NULL                           // Pointer to return value feb
          , jshep                          // Shepherd number
          , num_clone - 1                  // Number of instances - 1
          );

        assert( ret == QTHREAD_SUCCESS );
      }
    }
#endif

    driver_resize_worker_scratch( NULL );

    // Verify all workers allocated.

    bool ok = true;
    for ( int iwork = 0; ok && iwork < s_number_workers; ++iwork ) { ok = 0 != s_exec[iwork]; }

    if ( ! ok ) {
      std::ostringstream msg;
      msg << "Kokkos::Impl::QthreadsExec::resize : FAILED for workers {";
      for ( int iwork = 0; iwork < s_number_workers; ++iwork ) {
         if ( 0 == s_exec[iwork] ) { msg << " " << ( s_number_workers - ( iwork + 1 ) ); }
      }
      msg << " }";
      Kokkos::Impl::throw_runtime_exception( msg.str() );
    }
  }
}

void QthreadsExec::exec_all( Qthreads &, QthreadsExecFunctionPointer func, const void * arg )
{
  verify_is_process("QthreadsExec::exec_all(...)",true);

/*
  fprintf( stdout, "QthreadsExec::exec_all\n");
  fflush(stdout);
*/

  s_active_function     = func;
  s_active_function_arg = arg;

  // Need to query which shepherd this main 'process' is running.

  const int main_shep = qthread_shep();

#if 0
  for ( int jshep = 0, iwork = 0; jshep < s_number_shepherds; ++jshep ) {
    for ( int i = jshep != main_shep ? 0 : 1; i < s_number_workers_per_shepherd; ++i, ++iwork ) {
      qthread_fork_to( driver_exec_all, NULL, NULL, jshep );
    }
  }
#else
  // If this function is used before the 'qthreads.task_policy' unit test,
  // the 'qthreads.task_policy' unit test fails with a seg-fault within libqthread.so.
  for ( int jshep = 0; jshep < s_number_shepherds; ++jshep ) {
    const int num_clone = jshep != main_shep ? s_number_workers_per_shepherd : s_number_workers_per_shepherd - 1;

    if ( num_clone ) {
      const int ret = qthread_fork_clones_to_local_priority
        ( driver_exec_all   // Function
        , NULL              // Function data block
        , NULL              // Pointer to return value feb
        , jshep             // Shepherd number
        , num_clone - 1     // Number of instances - 1
        );

      assert(ret == QTHREAD_SUCCESS);
    }
  }
#endif

  driver_exec_all( NULL );

  s_active_function     = 0;
  s_active_function_arg = 0;
}

void * QthreadsExec::exec_all_reduce_result()
{
  return s_exec[0]->m_scratch_alloc;
}

} // namespace Impl

} // namespace Kokkos

namespace Kokkos {

namespace Impl {

QthreadsTeamPolicyMember::QthreadsTeamPolicyMember()
  : m_exec( **worker_exec() )
  , m_team_shared( 0, 0 )
  , m_team_size( 1 )
  , m_team_rank( 0 )
  , m_league_size( 1 )
  , m_league_end( 1 )
  , m_league_rank( 0 )
{
  m_exec.shared_reset( m_team_shared );
}

QthreadsTeamPolicyMember::QthreadsTeamPolicyMember( const QthreadsTeamPolicyMember::TaskTeam & )
  : m_exec( **worker_exec() )
  , m_team_shared( 0, 0 )
  , m_team_size( s_number_workers_per_shepherd )
  , m_team_rank( m_exec.shepherd_worker_rank() )
  , m_league_size( 1 )
  , m_league_end( 1 )
  , m_league_rank( 0 )
{
  m_exec.shared_reset( m_team_shared );
}

} // namespace Impl

} // namespace Kokkos

#else
void KOKKOS_SRC_QTHREADS_EXEC_PREVENT_LINK_ERROR() {}
#endif // #if defined( KOKKOS_ENABLE_QTHREADS )

