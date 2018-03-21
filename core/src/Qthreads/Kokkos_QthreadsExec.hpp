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

#ifndef KOKKOS_QTHREADSEXEC_HPP
#define KOKKOS_QTHREADSEXEC_HPP

#include <Kokkos_Macros.hpp>
#if defined( KOKKOS_ENABLE_QTHREADS )

#include <Kokkos_Qthreads.hpp>

#include <impl/Kokkos_Traits.hpp>
#include <impl/Kokkos_HostThreadTeam.hpp>

#include <Kokkos_Atomic.hpp>

#include <Kokkos_UniqueToken.hpp>

#include <impl/Kokkos_Spinwait.hpp>

//----------------------------------------------------------------------------

namespace Kokkos { namespace Impl {

class QthreadsExec;

typedef void (*QthreadsExecFunctionPointer)( QthreadsExec &, const void * );

extern int g_qthreads_hardware_max_threads;

extern __thread int t_qthreads_hardware_id;
extern __thread QthreadsExec * t_qthreads_instance;

//----------------------------------------------------------------------------
/** \brief  Data for Qthreads thread execution */

class QthreadsExec {
public:

  friend class Kokkos::Qthreads;

  enum { MAX_THREAD_COUNT = 512 };

  QthreadsExec( int arg_pool_size )
    : m_pool_size{ arg_pool_size }
    , m_level{ 1 }
  {
    Qthreads::memory_space space ;

    printf("allocating m_pool\n");
    void * const ptr = space.allocate( sizeof(Impl::QthreadsExec) );
    for(int i = 0; i < MAX_THREAD_COUNT; i++)
      m_pool[ i ] = new( ptr ) HostThreadTeamData();

    printf("allocated m_pool %p\n", &m_pool);
}
private:

  /*
  const int shepherd_rank        = qthread_shep();
  const int shepherd_worker_rank = qthread_worker_local( NULL );
  const int worker_rank          = shepherd_rank * s_number_workers_per_shepherd + shepherd_worker_rank;
  */
  /*
  QthreadsExec( int arg_pool_size )
    : m_pool_size{ arg_pool_size }
    , m_level{ 1 }
    , m_pool()
    , m_worker_base { s_exec }
    , m_shepherd_base { s_exec + s_number_workers_per_shepherd * ( ( s_number_shepherds - ( shepherd_rank + 1 ) ) }
    , m_scratch_alloc { ( (unsigned char *) this ) + s_base_size }
    , m_reduce_end { s_worker_reduce_end }
    , m_shepherd_rank { shepherd_rank }
    , m_shepherd_size { s_number_shepherds }
    , m_shepherd_worker_rank { shepherd_worker_rank }
    , m_shepherd_worker_size { s_number_workers_per_shepherd }
    , m_worker_rank          { worker_rank }
    , m_worker_size          { s_number_workers }
    , m_worker_state         { QthreadsExec::Active }
  {}
  */
  ~QthreadsExec();
  QthreadsExec( );
  QthreadsExec( const QthreadsExec & ) {
    /*
    Qthreads::memory_space space ;

    printf("allocating m_pool\n");
    void * const ptr = space.allocate( sizeof(Impl::QthreadsExec) );
    for(int i = 0; i < MAX_THREAD_COUNT; i++)
      m_pool[ i ] = new( ptr ) HostThreadTeamData();
    printf("allocated m_pool %p\n", &m_pool);
    */
  };
  QthreadsExec & operator = ( const QthreadsExec & );


  enum { Inactive = 0, Active = 1 };

  const QthreadsExec * const * m_worker_base;
  const QthreadsExec * const * m_shepherd_base;

  void  * m_scratch_alloc;  ///< Scratch memory [ reduce, team, shared ]
  int     m_reduce_end;     ///< End of scratch reduction memory

  int     m_shepherd_rank;
  int     m_shepherd_size;

  int     m_shepherd_worker_rank;
  int     m_shepherd_worker_size;

  /*
   *  m_worker_rank = m_shepherd_rank * m_shepherd_worker_size + m_shepherd_worker_rank
   *  m_worker_size = m_shepherd_size * m_shepherd_worker_size
   */
  int     m_worker_rank;
  int     m_worker_size;

  int mutable volatile m_worker_state;


  int m_pool_size;
  int m_level;

  HostThreadTeamData * m_pool[ MAX_THREAD_COUNT ];

public:
  static void verify_is_master( const char * const );
  void clear_thread_data();

  static void validate_partition( const int nthreads
                                  , int & num_partitions
                                  , int & partition_size
                                  );

  void resize_thread_data( size_t pool_reduce_bytes
                           , size_t team_reduce_bytes
                           , size_t team_shared_bytes
                           , size_t thread_local_bytes );

//  void resize_thread_data( size_t pool_reduce_bytes
//                            , size_t team_reduce_bytes
//                            , size_t team_shared_bytes
//                            , size_t thread_local_bytes )
//   {
//     std::cout << "resizing thread data" << std::endl;
//   const size_t member_bytes =
//     sizeof(int64_t) *
//     HostThreadTeamData::align_to_int64( sizeof(HostThreadTeamData) );

//   std::cout << "mpooling" << std::endl;
//   HostThreadTeamData * root = m_pool[0] ;

//   const size_t old_pool_reduce  = root ? root->pool_reduce_bytes() : 0 ;
//   const size_t old_team_reduce  = root ? root->team_reduce_bytes() : 0 ;
//   const size_t old_team_shared  = root ? root->team_shared_bytes() : 0 ;
//   const size_t old_thread_local = root ? root->thread_local_bytes() : 0 ;
//   const size_t old_alloc_bytes  = root ? ( member_bytes + root->scratch_bytes() ) : 0 ;

//   // Allocate if any of the old allocation is tool small:

//   std::cout << "allocating" << std::endl;
//   const bool allocate = ( old_pool_reduce  < pool_reduce_bytes ) ||
//                         ( old_team_reduce  < team_reduce_bytes ) ||
//                         ( old_team_shared  < team_shared_bytes ) ||
//                         ( old_thread_local < thread_local_bytes );

//   if ( allocate ) {

//     std::cout << "should allocate" << std::endl;
//     if ( pool_reduce_bytes < old_pool_reduce ) { pool_reduce_bytes = old_pool_reduce ; }
//     if ( team_reduce_bytes < old_team_reduce ) { team_reduce_bytes = old_team_reduce ; }
//     if ( team_shared_bytes < old_team_shared ) { team_shared_bytes = old_team_shared ; }
//     if ( thread_local_bytes < old_thread_local ) { thread_local_bytes = old_thread_local ; }

//     const size_t alloc_bytes =
//       member_bytes +
//       HostThreadTeamData::scratch_size( pool_reduce_bytes
//                                       , team_reduce_bytes
//                                       , team_shared_bytes
//                                       , thread_local_bytes );

//     Qthreads::memory_space space ;

//     memory_fence();

//     std::cout << "assigning scratch for pool of size " << m_pool_size << std::endl;
//     // how should qthreads do this?
//     //#pragma omp parallel num_threads(m_pool_size)
//     {

//       std::cout << "which rank am I?" << std::endl;
//       const int rank = 0; // omp_get_thread_num();

//       if ( 0 != m_pool[rank] ) {

//         m_pool[rank]->disband_pool();

//         space.deallocate( m_pool[rank] , old_alloc_bytes );
//       }

//       void * const ptr = space.allocate( alloc_bytes );

//       m_pool[ rank ] = new( ptr ) HostThreadTeamData();

//       std::cout << "assigning scratch" << std::endl;
//       m_pool[ rank ]->
//         scratch_assign( ((char *)ptr) + member_bytes
//                       , alloc_bytes
//                       , pool_reduce_bytes
//                       , team_reduce_bytes
//                       , team_shared_bytes
//                       , thread_local_bytes
//                       );

//       std::cout << "assigned scratch" << std::endl;
//       memory_fence();
//     }
// /* END #pragma omp parallel */

//     std::cout << "organizing pool" << std::endl;

//     std::cout << "m_pool " << m_pool << std::endl;
//     std::cout << "m_pool_size " << m_pool_size << std::endl;
//     HostThreadTeamData::organize_pool( m_pool , m_pool_size );
//     std::cout << "organized pool" << std::endl;
//   }



//   }

  inline
  HostThreadTeamData * get_thread_data() const noexcept
  { printf("m_pool %p\n", &m_pool); return m_pool[0]; }

  inline
  HostThreadTeamData * get_thread_data( int i ) const noexcept
  { return m_pool[i]; }

  /** Execute the input function on all available Qthreads workers. */
  static void exec_all( Qthreads &, QthreadsExecFunctionPointer, const void * );

  /** Barrier across all workers participating in the 'exec_all'. */
  void exec_all_barrier() const
  {
    const int rev_rank = m_worker_size - ( m_worker_rank + 1 );

    int n, j;

    for ( n = 1; ( ! ( rev_rank & n ) ) && ( ( j = rev_rank + n ) < m_worker_size ); n <<= 1 ) {
      Impl::spinwait_while_equal( m_worker_base[j]->m_worker_state, QthreadsExec::Active );
    }

    if ( rev_rank ) {
      m_worker_state = QthreadsExec::Inactive;
      Impl::spinwait_while_equal( m_worker_state, QthreadsExec::Inactive );
    }

    for ( n = 1; ( ! ( rev_rank & n ) ) && ( ( j = rev_rank + n ) < m_worker_size ); n <<= 1 ) {
      m_worker_base[j]->m_worker_state = QthreadsExec::Active;
    }
  }

  /** Barrier across workers within the shepherd with rank < team_rank. */
  void shepherd_barrier( const int team_size ) const
  {
    if ( m_shepherd_worker_rank < team_size ) {

      const int rev_rank = team_size - ( m_shepherd_worker_rank + 1 );

      int n, j;

      for ( n = 1; ( ! ( rev_rank & n ) ) && ( ( j = rev_rank + n ) < team_size ); n <<= 1 ) {
        Impl::spinwait_while_equal( m_shepherd_base[j]->m_worker_state, QthreadsExec::Active );
      }

      if ( rev_rank ) {
        m_worker_state = QthreadsExec::Inactive;
        Impl::spinwait_while_equal( m_worker_state, QthreadsExec::Inactive );
      }

      for ( n = 1; ( ! ( rev_rank & n ) ) && ( ( j = rev_rank + n ) < team_size ); n <<= 1 ) {
        m_shepherd_base[j]->m_worker_state = QthreadsExec::Active;
      }
    }
  }

  /** Reduce across all workers participating in the 'exec_all'. */
  template< class FunctorType, class ReducerType, class ArgTag >
  inline
  void exec_all_reduce( const FunctorType & func, const ReducerType & reduce ) const
  {
    typedef Kokkos::Impl::if_c< std::is_same<InvalidType, ReducerType>::value, FunctorType, ReducerType > ReducerConditional;
    typedef typename ReducerConditional::type ReducerTypeFwd;
    typedef Kokkos::Impl::FunctorValueJoin< ReducerTypeFwd, ArgTag > ValueJoin;

    const int rev_rank = m_worker_size - ( m_worker_rank + 1 );

    int n, j;

    for ( n = 1; ( ! ( rev_rank & n ) ) && ( ( j = rev_rank + n ) < m_worker_size ); n <<= 1 ) {
      const QthreadsExec & fan = *m_worker_base[j];

      Impl::spinwait_while_equal( fan.m_worker_state, QthreadsExec::Active );

      ValueJoin::join( ReducerConditional::select( func, reduce ), m_scratch_alloc, fan.m_scratch_alloc );
    }

    if ( rev_rank ) {
      m_worker_state = QthreadsExec::Inactive;
      Impl::spinwait_while_equal( m_worker_state, QthreadsExec::Inactive );
    }

    for ( n = 1; ( ! ( rev_rank & n ) ) && ( ( j = rev_rank + n ) < m_worker_size ); n <<= 1 ) {
      m_worker_base[j]->m_worker_state = QthreadsExec::Active;
    }
  }

  /** Scan across all workers participating in the 'exec_all'. */
  template< class FunctorType, class ArgTag >
  inline
  void exec_all_scan( const FunctorType & func ) const
  {
    typedef Kokkos::Impl::FunctorValueInit< FunctorType, ArgTag > ValueInit;
    typedef Kokkos::Impl::FunctorValueJoin< FunctorType, ArgTag > ValueJoin;
    typedef Kokkos::Impl::FunctorValueOps<  FunctorType, ArgTag > ValueOps;

    const int rev_rank = m_worker_size - ( m_worker_rank + 1 );

    int n, j;

    for ( n = 1; ( ! ( rev_rank & n ) ) && ( ( j = rev_rank + n ) < m_worker_size ); n <<= 1 ) {
      Impl::spinwait_while_equal( m_worker_base[j]->m_worker_state, QthreadsExec::Active );
    }

    if ( rev_rank ) {
      m_worker_state = QthreadsExec::Inactive;
      Impl::spinwait_while_equal( m_worker_state, QthreadsExec::Inactive );
    }
    else {
      // Root thread scans across values before releasing threads.
      // Worker data is in reverse order, so m_worker_base[0] is the
      // highest ranking thread.

      // Copy from lower ranking to higher ranking worker.
      for ( int i = 1; i < m_worker_size; ++i ) {
        ValueOps::copy( func
                      , m_worker_base[i-1]->m_scratch_alloc
                      , m_worker_base[i]->m_scratch_alloc
                      );
      }

      ValueInit::init( func, m_worker_base[m_worker_size-1]->m_scratch_alloc );

      // Join from lower ranking to higher ranking worker.
      // Value at m_worker_base[n-1] is zero so skip adding it to m_worker_base[n-2].
      for ( int i = m_worker_size - 1; --i > 0; ) {
        ValueJoin::join( func, m_worker_base[i-1]->m_scratch_alloc, m_worker_base[i]->m_scratch_alloc );
      }
    }

    for ( n = 1; ( ! ( rev_rank & n ) ) && ( ( j = rev_rank + n ) < m_worker_size ); n <<= 1 ) {
      m_worker_base[j]->m_worker_state = QthreadsExec::Active;
    }
  }

  //----------------------------------------

  template< class Type >
  inline
  volatile Type * shepherd_team_scratch_value() const
  { return (volatile Type*)( ( (unsigned char *) m_scratch_alloc ) + m_reduce_end ); }

  template< class Type >
  inline
  void shepherd_broadcast( Type & value, const int team_size, const int team_rank ) const
  {
    if ( m_shepherd_base ) {
      Type * const shared_value = m_shepherd_base[0]->shepherd_team_scratch_value<Type>();
      if ( m_shepherd_worker_rank == team_rank ) { *shared_value = value; }
      memory_fence();
      shepherd_barrier( team_size );
      value = *shared_value;
    }
  }

  template< class Type >
  inline
  Type shepherd_reduce( const int team_size, const Type & value ) const
  {
    volatile Type * const shared_value = shepherd_team_scratch_value<Type>();
    *shared_value = value;
//    *shepherd_team_scratch_value<Type>() = value;

    memory_fence();

    const int rev_rank = team_size - ( m_shepherd_worker_rank + 1 );

    int n, j;

    for ( n = 1; ( ! ( rev_rank & n ) ) && ( ( j = rev_rank + n ) < team_size ); n <<= 1 ) {
      Impl::spinwait_while_equal( m_shepherd_base[j]->m_worker_state, QthreadsExec::Active );
    }

    if ( rev_rank ) {
      m_worker_state = QthreadsExec::Inactive;
      Impl::spinwait_while_equal( m_worker_state, QthreadsExec::Inactive );
    }
    else {
      Type & accum = *m_shepherd_base[0]->shepherd_team_scratch_value<Type>();
      for ( int i = 1; i < n; ++i ) {
        accum += *m_shepherd_base[i]->shepherd_team_scratch_value<Type>();
      }
      for ( int i = 1; i < n; ++i ) {
        *m_shepherd_base[i]->shepherd_team_scratch_value<Type>() = accum;
      }

      memory_fence();
    }

    for ( n = 1; ( ! ( rev_rank & n ) ) && ( ( j = rev_rank + n ) < team_size ); n <<= 1 ) {
      m_shepherd_base[j]->m_worker_state = QthreadsExec::Active;
    }

    return *shepherd_team_scratch_value<Type>();
  }

  template< class JoinOp >
  inline
  typename JoinOp::value_type
  shepherd_reduce( const int team_size
                 , const typename JoinOp::value_type & value
                 , const JoinOp & op ) const
  {
    typedef typename JoinOp::value_type Type;

    volatile Type * const shared_value = shepherd_team_scratch_value<Type>();
    *shared_value = value;
//    *shepherd_team_scratch_value<Type>() = value;

    memory_fence();

    const int rev_rank = team_size - ( m_shepherd_worker_rank + 1 );

    int n, j;

    for ( n = 1; ( ! ( rev_rank & n ) ) && ( ( j = rev_rank + n ) < team_size ); n <<= 1 ) {
      Impl::spinwait_while_equal( m_shepherd_base[j]->m_worker_state, QthreadsExec::Active );
    }

    if ( rev_rank ) {
      m_worker_state = QthreadsExec::Inactive;
      Impl::spinwait_while_equal( m_worker_state, QthreadsExec::Inactive );
    }
    else {
      volatile Type & accum = *m_shepherd_base[0]->shepherd_team_scratch_value<Type>();
      for ( int i = 1; i < team_size; ++i ) {
        op.join( accum, *m_shepherd_base[i]->shepherd_team_scratch_value<Type>() );
      }
      for ( int i = 1; i < team_size; ++i ) {
        *m_shepherd_base[i]->shepherd_team_scratch_value<Type>() = accum;
      }

      memory_fence();
    }

    for ( n = 1; ( ! ( rev_rank & n ) ) && ( ( j = rev_rank + n ) < team_size ); n <<= 1 ) {
      m_shepherd_base[j]->m_worker_state = QthreadsExec::Active;
    }

    return *shepherd_team_scratch_value<Type>();
  }

  template< class Type >
  inline
  Type shepherd_scan( const int team_size
                    , const Type & value
                    ,       Type * const global_value = 0 ) const
  {
    *shepherd_team_scratch_value<Type>() = value;

    memory_fence();

    const int rev_rank = team_size - ( m_shepherd_worker_rank + 1 );

    int n, j;

    for ( n = 1; ( ! ( rev_rank & n ) ) && ( ( j = rev_rank + n ) < team_size ); n <<= 1 ) {
      Impl::spinwait_while_equal( m_shepherd_base[j]->m_worker_state, QthreadsExec::Active );
    }

    if ( rev_rank ) {
      m_worker_state = QthreadsExec::Inactive;
      Impl::spinwait_while_equal( m_worker_state, QthreadsExec::Inactive );
    }
    else {
      // Root thread scans across values before releasing threads.
      // Worker data is in reverse order, so m_shepherd_base[0] is the
      // highest ranking thread.

      // Copy from lower ranking to higher ranking worker.

      Type accum = *m_shepherd_base[0]->shepherd_team_scratch_value<Type>();
      for ( int i = 1; i < team_size; ++i ) {
        const Type tmp = *m_shepherd_base[i]->shepherd_team_scratch_value<Type>();
        accum += tmp;
        *m_shepherd_base[i-1]->shepherd_team_scratch_value<Type>() = tmp;
      }

      *m_shepherd_base[team_size-1]->shepherd_team_scratch_value<Type>() =
        global_value ? atomic_fetch_add( global_value, accum ) : 0;

      // Join from lower ranking to higher ranking worker.
      for ( int i = team_size; --i; ) {
        *m_shepherd_base[i-1]->shepherd_team_scratch_value<Type>() += *m_shepherd_base[i]->shepherd_team_scratch_value<Type>();
      }

      memory_fence();
    }

    for ( n = 1; ( ! ( rev_rank & n ) ) && ( ( j = rev_rank + n ) < team_size ); n <<= 1 ) {
      m_shepherd_base[j]->m_worker_state = QthreadsExec::Active;
    }

    return *shepherd_team_scratch_value<Type>();
  }

  //----------------------------------------

  static inline
  int align_alloc( int size )
  {
    enum { ALLOC_GRAIN = 1 << 6 /* power of two, 64bytes */ };
    enum { ALLOC_GRAIN_MASK = ALLOC_GRAIN - 1 };
    return ( size + ALLOC_GRAIN_MASK ) & ~ALLOC_GRAIN_MASK;
  }

  void shared_reset( Qthreads::scratch_memory_space & );

  void * exec_all_reduce_value() const { return m_scratch_alloc; }

  static void * exec_all_reduce_result();

  static void resize_worker_scratch( const int reduce_size, const int shared_size );
  static void clear_workers();

  //----------------------------------------

  inline int worker_rank() const { return m_worker_rank; }
  inline int worker_size() const { return m_worker_size; }
  inline int shepherd_worker_rank() const { return m_shepherd_worker_rank; }
  inline int shepherd_worker_size() const { return m_shepherd_worker_size; }
  inline int shepherd_rank() const { return m_shepherd_rank; }
  inline int shepherd_size() const { return m_shepherd_size; }

  static int worker_per_shepherd();
};

} // namespace Impl

} // namespace Kokkos

//----------------------------------------------------------------------------

namespace Kokkos {

namespace Impl {

class QthreadsTeamPolicyMember {
private:
  typedef Kokkos::Qthreads                       execution_space;
  typedef execution_space::scratch_memory_space  scratch_memory_space;

  Impl::QthreadsExec   & m_exec;
  scratch_memory_space   m_team_shared;
  const int              m_team_size;
  const int              m_team_rank;
  const int              m_league_size;
  const int              m_league_end;
        int              m_league_rank;

public:
  KOKKOS_INLINE_FUNCTION
  const scratch_memory_space & team_shmem() const { return m_team_shared; }

  KOKKOS_INLINE_FUNCTION int league_rank() const { return m_league_rank; }
  KOKKOS_INLINE_FUNCTION int league_size() const { return m_league_size; }
  KOKKOS_INLINE_FUNCTION int team_rank() const { return m_team_rank; }
  KOKKOS_INLINE_FUNCTION int team_size() const { return m_team_size; }

  KOKKOS_INLINE_FUNCTION void team_barrier() const
#if ! defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )
  {}
#else
  { m_exec.shepherd_barrier( m_team_size ); }
#endif

  template< typename Type >
  KOKKOS_INLINE_FUNCTION Type team_broadcast( const Type & value, int rank ) const
#if ! defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )
  { return Type(); }
#else
  { return m_exec.template shepherd_broadcast<Type>( value, m_team_size, rank ); }
#endif

  template< typename Type >
  KOKKOS_INLINE_FUNCTION Type team_reduce( const Type & value ) const
#if ! defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )
  { return Type(); }
#else
  { return m_exec.template shepherd_reduce<Type>( m_team_size, value ); }
#endif

  template< typename JoinOp >
  KOKKOS_INLINE_FUNCTION typename JoinOp::value_type
  team_reduce( const typename JoinOp::value_type & value
             , const JoinOp & op ) const
#if ! defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )
  { return typename JoinOp::value_type(); }
#else
  { return m_exec.template shepherd_reduce<JoinOp>( m_team_size, value, op ); }
#endif

  /** \brief  Intra-team exclusive prefix sum with team_rank() ordering.
   *
   *  The highest rank thread can compute the reduction total as
   *    reduction_total = dev.team_scan( value ) + value;
   */
  template< typename Type >
  KOKKOS_INLINE_FUNCTION Type team_scan( const Type & value ) const
#if ! defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )
  { return Type(); }
#else
  { return m_exec.template shepherd_scan<Type>( m_team_size, value ); }
#endif

  /** \brief  Intra-team exclusive prefix sum with team_rank() ordering
   *          with intra-team non-deterministic ordering accumulation.
   *
   *  The global inter-team accumulation value will, at the end of the league's
   *  parallel execution, be the scan's total.  Parallel execution ordering of
   *  the league's teams is non-deterministic.  As such the base value for each
   *  team's scan operation is similarly non-deterministic.
   */
  template< typename Type >
  KOKKOS_INLINE_FUNCTION Type team_scan( const Type & value, Type * const global_accum ) const
#if ! defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )
  { return Type(); }
#else
  { return m_exec.template shepherd_scan<Type>( m_team_size, value, global_accum ); }
#endif

  //----------------------------------------
  // Private driver for task-team parallel.

  struct TaskTeam {};

  QthreadsTeamPolicyMember();
  explicit QthreadsTeamPolicyMember( const TaskTeam & );

  //----------------------------------------
  // Private for the driver ( for ( member_type i( exec, team ); i; i.next_team() ) { ... }

  // Initialize.
  template< class ... Properties >
  QthreadsTeamPolicyMember( Impl::QthreadsExec & exec
                          , const Kokkos::Impl::TeamPolicyInternal< Qthreads, Properties... > & team )
    : m_exec( exec )
    , m_team_shared( 0, 0 )
    , m_team_size( team.m_team_size )
    , m_team_rank( exec.shepherd_worker_rank() )
    , m_league_size( team.m_league_size )
    , m_league_end( team.m_league_size - team.m_shepherd_iter * ( exec.shepherd_size() - ( exec.shepherd_rank() + 1 ) ) )
    , m_league_rank( m_league_end > team.m_shepherd_iter ? m_league_end - team.m_shepherd_iter : 0 )
  {
    m_exec.shared_reset( m_team_shared );
  }

  // Continue.
  operator bool () const { return m_league_rank < m_league_end; }

  // Iterate.
  void next_team() { ++m_league_rank; m_exec.shared_reset( m_team_shared ); }
};
} // namespace Impl

} // namespace Kokkos

namespace Kokkos {
#if !defined( KOKKOS_DISABLE_DEPRECATED )

inline
int Qthreads::thread_pool_size( int depth )
{
  std::cout << "depth " << depth << std::endl;
  std::cout << "qthread_num_shepards " << qthread_num_shepherds << " qthread_num_workers " << qthread_num_workers() << std::endl;
  return depth < 2
                 ? qthread_num_shepherds()*qthread_num_workers()
                 : 1;
}

KOKKOS_INLINE_FUNCTION
int Qthreads::hardware_thread_id() noexcept
{
#if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )
  return Impl::t_qthreads_hardware_id;
#else
  return -1 ;
#endif
}

inline
int Qthreads::max_hardware_threads() noexcept
{
  return Impl::g_qthreads_hardware_max_threads;
}

#endif // KOKKOS_DISABLE_DEPRECATED

} // namespace Kokkos


#endif
#endif // #define KOKKOS_QTHREADSEXEC_HPP

