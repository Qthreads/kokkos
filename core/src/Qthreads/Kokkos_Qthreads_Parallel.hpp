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

#ifndef KOKKOS_QTHREADS_PARALLEL_HPP
#define KOKKOS_QTHREADS_PARALLEL_HPP

#include <Kokkos_Macros.hpp>
#if defined( KOKKOS_ENABLE_QTHREADS )

#include <vector>

#include <Qthreads/Kokkos_QthreadsExec.hpp>
#include <impl/Kokkos_FunctorAdapter.hpp>

#include <KokkosExp_MDRangePolicy.hpp>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

template< class FunctorType , class ... Traits >
class ParallelFor< FunctorType
                 , Kokkos::RangePolicy< Traits ... >
                 , Kokkos::Qthreads
                 >
{
private:

  typedef Kokkos::RangePolicy< Traits ... >  Policy ;
  typedef typename Policy::work_tag     WorkTag ;
  typedef typename Policy::WorkRange    WorkRange ;
  typedef typename Policy::member_type  Member ;

        QthreadsExec   * m_instance;
  const FunctorType    m_functor;
  const Policy         m_policy;

  template< class TagType >
  inline static
  typename std::enable_if< std::is_same< TagType , void >::value >::type
  exec_range( const FunctorType & functor
              , const Member ibeg , const Member iend )
    {
       #ifdef KOKKOS_ENABLE_AGGRESSIVE_VECTORIZATION
       #ifdef KOKKOS_ENABLE_PRAGMA_IVDEP
       #pragma ivdep
       #endif
       #endif
      std::cout << " exec_range 1 " << std::endl;
      for ( Member iwork = ibeg ; iwork < iend ; ++iwork ) {
        functor( iwork );
      }
    }

  template< class TagType >
  inline static
  typename std::enable_if< ! std::is_same< TagType , void >::value >::type
  exec_range( const FunctorType & functor
            , const Member ibeg , const Member iend )
    {
      const TagType t{} ;
      #ifdef KOKKOS_ENABLE_AGGRESSIVE_VECTORIZATION
      #ifdef KOKKOS_ENABLE_PRAGMA_IVDEP
      #pragma ivdep
      #endif
      #endif
      std::cout << "exec_range 1.1" << std::endl;
      for ( Member iwork = ibeg ; iwork < iend ; ++iwork ) {
        functor( t , iwork );
      }
    }

public:

  inline void execute() const
  {
    std::cout << "executing " << std::endl;
    enum { is_dynamic = std::is_same< typename Policy::schedule_type::type
         , Kokkos::Dynamic >::value
         };

    if ( Qthreads::in_parallel() ) {
      std::cout << "in parallel do range" << std::endl;
      exec_range< WorkTag >( m_functor
                           , m_policy.begin()
                           , m_policy.end() );
    }
    else {

      std::cout << "is master?" << std::endl;
      QthreadsExec::verify_is_master("Kokkos::Qthreads parallel_for");

      std::cout << "pool size" << std::endl;
      const int pool_size = Qthreads::thread_pool_size();
      std::cout << "pool sized " << pool_size << std::endl;

      //#pragma omp parallel num_threads(pool_size)
      {
        std::cout << "t_qthreads_instance" << t_qthreads_instance << std::endl;
        std::cout << "m_instance" << m_instance << std::endl;
        HostThreadTeamData & data = *(m_instance->get_thread_data());
        std::cout << "partitioning " << std::endl;
        data.set_work_partition( m_policy.end() - m_policy.begin()
            , m_policy.chunk_size() );

        std::cout << "waiting " << std::endl;
        if ( is_dynamic ) {
          // Make sure work partition is set before stealing
          if ( data.pool_rendezvous() ) data.pool_rendezvous_release();
        }

        std::pair<int64_t,int64_t> range(0,0);

        std::cout << "stealing" << std::endl;
        do {
          std::cout << "do whiling" << std::endl;
          range = is_dynamic ? data.get_work_stealing_chunk()
            : data.get_work_partition();

          ParallelFor::template
            exec_range< WorkTag >( m_functor
                , range.first  + m_policy.begin()
                , range.second + m_policy.begin() );

        } while ( is_dynamic && 0 <= range.first );
      }
    }
    std::cout << "done" << std::endl;
  }

  inline
  ParallelFor( const FunctorType & arg_functor
             , Policy arg_policy )
    : m_instance( t_qthreads_instance )
    , m_functor( arg_functor )
    , m_policy(  arg_policy )
    { std::cout << "commented out constructor" << std::endl; }
};

// MDRangePolicy impl
template< class FunctorType , class ... Traits >
class ParallelFor< FunctorType
                 , Kokkos::Experimental::MDRangePolicy< Traits ... >
                 , Kokkos::Qthreads
                 >
{
private:

  typedef Kokkos::Experimental::MDRangePolicy< Traits ... > MDRangePolicy ;
  typedef typename MDRangePolicy::impl_range_policy         Policy ;
  typedef typename MDRangePolicy::work_tag                  WorkTag ;

  typedef typename Policy::WorkRange    WorkRange ;
  typedef typename Policy::member_type  Member ;

  typedef typename Kokkos::Experimental::Impl::HostIterateTile< MDRangePolicy, FunctorType, typename MDRangePolicy::work_tag, void > iterate_type;

        QthreadsExec   * m_instance ;
  const FunctorType   m_functor ;
  const MDRangePolicy m_mdr_policy ;
  const Policy        m_policy ;  // construct as RangePolicy( 0, num_tiles ).set_chunk_size(1) in ctor

  inline static
  void
  exec_range( const MDRangePolicy & mdr_policy
            , const FunctorType & functor
            , const Member ibeg , const Member iend )
    {
      #ifdef KOKKOS_ENABLE_AGGRESSIVE_VECTORIZATION
      #ifdef KOKKOS_ENABLE_PRAGMA_IVDEP
      #pragma ivdep
      #endif
      #endif
      std::cout << "exec_range experimental" << std::endl;
      for ( Member iwork = ibeg ; iwork < iend ; ++iwork ) {
        iterate_type( mdr_policy, functor )( iwork );
      }
    }

public:

  inline void execute() const
  {
    std::cout << "experimental execute" << std::endl;
      enum { is_dynamic = std::is_same< typename Policy::schedule_type::type
                                      , Kokkos::Dynamic >::value };

    if ( Qthreads::in_parallel() ) {
      ParallelFor::exec_range ( m_mdr_policy
                              , m_functor
                              , m_policy.begin()
                              , m_policy.end() );
    }
    else {

      QthreadsExec::verify_is_master("Kokkos::Qthreads parallel_for");

      const int pool_size = Qthreads::thread_pool_size();
      //#pragma omp parallel num_threads(pool_size)
      auto pfor = [&] {
        // mark as appropriate variables volatile or use memory fences 
        HostThreadTeamData & data = *(m_instance->get_thread_data());

        data.set_work_partition( m_policy.end() - m_policy.begin()
                               , m_policy.chunk_size() );

        if ( is_dynamic ) {
          // Make sure work partition is set before stealing
          if ( data.pool_rendezvous() ) data.pool_rendezvous_release();
        }

        std::pair<int64_t,int64_t> range(0,0);

        do {

          range = is_dynamic ? data.get_work_stealing_chunk()
                             : data.get_work_partition();

          ParallelFor::exec_range( m_mdr_policy
                                 , m_functor
                                 , range.first  + m_policy.begin()
                                 , range.second + m_policy.begin() );

        } while ( is_dynamic && 0 <= range.first );
      };
      for(int i = 0; i < pool_size; ++i) {
        qthread_fork_to(&pfor, NULL, NULL, i);
      }

    }
  }

  inline
  ParallelFor( const FunctorType & arg_functor
             , MDRangePolicy arg_policy )
    : m_instance( t_qthreads_instance )
    , m_functor( arg_functor )
    , m_mdr_policy( arg_policy )
    , m_policy( Policy(0, m_mdr_policy.m_num_tiles).set_chunk_size(1) )
    { std::cout << "MDRange parallel for!" << std::endl;     }
};

} // namespace Impl
} // namespace Kokkos
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

template< class FunctorType , class ReducerType , class ... Traits>
class ParallelReduce< FunctorType
                      , Kokkos::RangePolicy< Traits... >
                    , ReducerType
                    , Kokkos::Qthreads
                    >
{
private:

  typedef Kokkos::RangePolicy< Traits ... > Policy ;

  typedef typename Policy::work_tag     WorkTag ;
  typedef typename Policy::WorkRange    WorkRange ;
  typedef typename Policy::member_type  Member ;

  typedef FunctorAnalysis< FunctorPatternInterface::REDUCE , Policy , FunctorType > Analysis ;

  typedef Kokkos::Impl::if_c< std::is_same<InvalidType,ReducerType>::value, FunctorType, ReducerType> ReducerConditional;
  typedef typename ReducerConditional::type ReducerTypeFwd;


  // Static Assert WorkTag void if ReducerType not InvalidType

  typedef Kokkos::Impl::FunctorValueInit<   ReducerTypeFwd, WorkTag > ValueInit ;
  typedef Kokkos::Impl::FunctorValueJoin<   ReducerTypeFwd, WorkTag > ValueJoin ;

  typedef typename Analysis::pointer_type    pointer_type ;
  typedef typename Analysis::reference_type  reference_type ;

        QthreadsExec   * m_instance;
  const FunctorType    m_functor;
  const Policy         m_policy;
  const ReducerType    m_reducer;
  const pointer_type   m_result_ptr;

  template< class TagType >
  inline static
  typename std::enable_if< std::is_same< TagType , void >::value >::type
  exec_range( const FunctorType & functor
              , const Member ibeg , const Member iend
              , reference_type update )
  {
    for ( Member iwork = ibeg ; iwork < iend ; ++iwork ) {
      functor( iwork , update );
    }
  }

  template< class TagType >
  inline static
  typename std::enable_if< ! std::is_same< TagType , void >::value >::type
  exec_range( const FunctorType & functor
              , const Member ibeg , const Member iend
              , reference_type update )
  {
    const TagType t{} ;
    for ( Member iwork = ibeg ; iwork < iend ; ++iwork ) {
      functor( t , iwork , update );
    }
  }

public:

  inline void execute() const
    {
            enum { is_dynamic = std::is_same< typename Policy::schedule_type::type
                                      , Kokkos::Dynamic >::value };

      QthreadsExec::verify_is_master("Kokkos::Qthreads parallel_reduce");
      std::cout << "getting value size" << std::endl;
      const size_t pool_reduce_bytes =
        Analysis::value_size( ReducerConditional::select(m_functor, m_reducer));

      std::cout << "resize thread data" << std::endl;
      m_instance->resize_thread_data( pool_reduce_bytes
                                    , 0 //  team_reduce_bytes
                                    , 0 // team_shared_bytes
                                    , 0 // thread_local_bytes
                                    );

      std::cout << "resized thread data" << std::endl;
      const int pool_size = Qthreads::thread_pool_size();
      std::cout << "got pool size" << std::endl;
      // need to do this in parallel. so I understand what it is doing now.
      // try this as a lambda. 
      //#pragma omp parallel num_threads(pool_size)
      auto red = [&]{
        // XXX: fix this, does it go in the constructor? 
        std::cout << "getting thread data" << std::endl;

        HostThreadTeamData & data = *(m_instance->get_thread_data());

        std::cout << "set work partition" << std::endl;
        data.set_work_partition( m_policy.end() - m_policy.begin()
                               , m_policy.chunk_size() );

        if ( is_dynamic ) {
          std::cout << "we are dynamic" << std::endl;
          // Make sure work partition is set before stealing
          if ( data.pool_rendezvous() ) data.pool_rendezvous_release();
        }

        std::cout << "updating" << std::endl;

        ReducerConditional::select(m_functor , m_reducer);

        std::cout << "done selecting" << std::endl;
        reference_type update =
          ValueInit::init( ReducerConditional::select(m_functor , m_reducer)
                           , data.pool_reduce_local() );
        std::cout << "reduced local" << std::endl;
        std::pair<int64_t,int64_t> range(0,0);

        do {

          std::cout << "reduce stealing" << std::endl;
          range = is_dynamic ? data.get_work_stealing_chunk()
                             : data.get_work_partition();

          std::cout << "range execing" << std::endl;
          ParallelReduce::template
            exec_range< WorkTag >( m_functor
                                 , range.first  + m_policy.begin()
                                 , range.second + m_policy.begin()
                                 , update );

        } while ( is_dynamic && 0 <= range.first );
        std::cout << "done" << std::endl;
      };
      for(int i = 0; i < pool_size; i++)
        qthread_fork_to( (qthread_f)&red, NULL, NULL, i ); 
      // Reduction:

      const pointer_type ptr = pointer_type( m_instance->get_thread_data(0)->pool_reduce_local() );
      printf("m_instance->get_thread_data(0) %p\n", m_instance->get_thread_data(0));
      printf("ptr %p\n", ptr);
      printf("ptr[0] %p\n", ptr[0]);
      for ( int i = 1 ; i < pool_size ; ++i ) {
        ValueJoin::join( ReducerConditional::select(m_functor , m_reducer)
                       , ptr
                       , m_instance->get_thread_data(i)->pool_reduce_local() );
      }

      Kokkos::Impl::FunctorFinal<  ReducerTypeFwd , WorkTag >::final( ReducerConditional::select(m_functor , m_reducer) , ptr );

      if ( m_result_ptr ) {
        const int n = Analysis::value_count( ReducerConditional::select(m_functor , m_reducer) );

        for ( int j = 0 ; j < n ; ++j ) {
          printf("m_result_ptr[j] %p\n", m_result_ptr[j]);
          printf("ptr %p\n", ptr);
          m_result_ptr[j] = ptr[j] ;
        }
      }

      // std::cout << "I'm not doing anything!!!" << std::endl;
      //  QthreadsExec::resize_worker_scratch
      //    ( /* reduction   memory */ ValueTraits::value_size( ReducerConditional::select(m_functor , m_reducer) )
      //    , /* team shared memory */ FunctorTeamShmemSize< FunctorType >::value( m_functor , m_policy.team_size() ) );

      //  Impl::QthreadsExec::exec_all( Qthreads::instance() , & ParallelReduce::exec , this );

      //  const pointer_type data = (pointer_type) QthreadsExec::exec_all_reduce_result();

      //  Kokkos::Impl::FunctorFinal< ReducerTypeFwd , WorkTag >::final( ReducerConditional::select(m_functor , m_reducer), data );

      //  if ( m_result_ptr ) {
      //    const unsigned n = ValueTraits::value_count( ReducerConditional::select(m_functor , m_reducer) );
      //    for ( unsigned i = 0 ; i < n ; ++i ) { m_result_ptr[i] = data[i]; }
      //  }
    }

  template< class ViewType >
  ParallelReduce( const FunctorType & arg_functor
                , const Policy      & arg_policy
                , const ViewType    & arg_result
                , typename std::enable_if<Kokkos::is_view< ViewType >::value &&
                                          !Kokkos::is_reducer_type< ReducerType >::value
                                          , void*>::type = NULL)
    : m_instance( t_qthreads_instance )
    , m_functor( arg_functor )
    , m_policy( arg_policy )
    , m_reducer( InvalidType() )
    , m_result_ptr( arg_result.ptr_on_device() )
  { std::cout << "constructing with nothing" << std::endl; }

  inline
  ParallelReduce( const FunctorType & arg_functor
                , Policy       arg_policy
                , const ReducerType& reducer )
  : m_functor( arg_functor )
  , m_policy( arg_policy )
  , m_reducer( reducer )
  , m_result_ptr( reducer.view().data() )
  { }
};


// MDRangePolicy impl
template< class FunctorType , class ReducerType , class ... Traits >
class ParallelReduce< FunctorType
                      , Kokkos::Experimental::MDRangePolicy< Traits ... >
                    , ReducerType
                    , Kokkos::Qthreads
                    >
{
private:

  typedef Kokkos::Experimental::MDRangePolicy< Traits ... > MDRangePolicy ;
  typedef typename MDRangePolicy::impl_range_policy         Policy ;

  typedef typename MDRangePolicy::work_tag                  WorkTag ;
  typedef typename Policy::WorkRange                        WorkRange ;
  typedef typename Policy::member_type                      Member ;

  typedef FunctorAnalysis< FunctorPatternInterface::REDUCE , Policy , FunctorType > Analysis ;

  typedef Kokkos::Impl::if_c< std::is_same<InvalidType,ReducerType>::value, FunctorType, ReducerType> ReducerConditional;
  typedef typename ReducerConditional::type ReducerTypeFwd;

  typedef typename ReducerTypeFwd::value_type ValueType;

  typedef Kokkos::Impl::FunctorValueInit<   ReducerTypeFwd, WorkTag > ValueInit ;
  typedef Kokkos::Impl::FunctorValueJoin<   ReducerTypeFwd, WorkTag > ValueJoin ;

  typedef typename Analysis::pointer_type    pointer_type ;
  typedef typename Analysis::reference_type  reference_type ;

  using iterate_type = typename Kokkos::Experimental::Impl::HostIterateTile< MDRangePolicy
                                                                           , FunctorType
                                                                           , WorkTag
                                                                           , ValueType
                                                                           >;

        QthreadsExec   * m_instance ;
  const FunctorType   m_functor ;
  const MDRangePolicy m_mdr_policy ;
  const Policy        m_policy ;     // construct as RangePolicy( 0, num_tiles ).set_chunk_size(1) in ctor
  const ReducerType   m_reducer ;
  const pointer_type  m_result_ptr ;

  inline static
  void
  exec_range( const MDRangePolicy & mdr_policy
            , const FunctorType & functor
            , const Member ibeg , const Member iend
            , reference_type update )
    {
      for ( Member iwork = ibeg ; iwork < iend ; ++iwork ) {
        iterate_type( mdr_policy, functor, update )( iwork );
      }
    }

public:

  inline void execute() const
    {
            enum { is_dynamic = std::is_same< typename Policy::schedule_type::type
                                      , Kokkos::Dynamic >::value };

      QthreadsExec::verify_is_master("Kokkos::Qthreads parallel_reduce");

      const size_t pool_reduce_bytes =
        Analysis::value_size( ReducerConditional::select(m_functor, m_reducer));

      m_instance->resize_thread_data( pool_reduce_bytes
                                    , 0 // team_reduce_bytes
                                    , 0 // team_shared_bytes
                                    , 0 // thread_local_bytes
                                    );

      const int pool_size = Qthreads::thread_pool_size();
      //#pragma omp parallel num_threads(pool_size)
      {
        HostThreadTeamData & data = *(m_instance->get_thread_data());

        data.set_work_partition( m_policy.end() - m_policy.begin()
                               , m_policy.chunk_size() );

        if ( is_dynamic ) {
          // Make sure work partition is set before stealing
          if ( data.pool_rendezvous() ) data.pool_rendezvous_release();
        }

        reference_type update =
          ValueInit::init( ReducerConditional::select(m_functor , m_reducer)
                         , data.pool_reduce_local() );

        std::pair<int64_t,int64_t> range(0,0);

        do {

          range = is_dynamic ? data.get_work_stealing_chunk()
                             : data.get_work_partition();

          ParallelReduce::exec_range ( m_mdr_policy, m_functor
                                     , range.first  + m_policy.begin()
                                     , range.second + m_policy.begin()
                                     , update );

        } while ( is_dynamic && 0 <= range.first );
      }
// END #pragma omp parallel

      // Reduction:

      const pointer_type ptr = pointer_type( m_instance->get_thread_data(0)->pool_reduce_local() );

      for ( int i = 1 ; i < pool_size ; ++i ) {
        ValueJoin::join( ReducerConditional::select(m_functor , m_reducer)
                       , ptr
                       , m_instance->get_thread_data(i)->pool_reduce_local() );
      }

      Kokkos::Impl::FunctorFinal<  ReducerTypeFwd , WorkTag >::final( ReducerConditional::select(m_functor , m_reducer) , ptr );

      if ( m_result_ptr ) {
        const int n = Analysis::value_count( ReducerConditional::select(m_functor , m_reducer) );

        for ( int j = 0 ; j < n ; ++j ) { m_result_ptr[j] = ptr[j] ; }
      }

      //QthreadsExec::resize_worker_scratch( ValueTraits::value_size( ReducerConditional::select(m_functor , m_reducer) ) , 0 );
      //Impl::QthreadsExec::exec_all( Qthreads::instance() , & ParallelReduce::exec , this );

      //const pointer_type data = (pointer_type) QthreadsExec::exec_all_reduce_result();

      //Kokkos::Impl::FunctorFinal< ReducerTypeFwd , WorkTag >::final( ReducerConditional::select(m_functor , m_reducer) , data );

      // if ( m_result_ptr ) {
      //   const unsigned n = ValueTraits::value_count( ReducerConditional::select(m_functor , m_reducer) );
      //   for ( unsigned i = 0 ; i < n ; ++i ) { m_result_ptr[i] = data[i]; }
      // }
    }

  template< class ViewType >
  inline
  ParallelReduce( const FunctorType  & arg_functor
                , const Policy       & arg_policy
                , const ViewType & arg_view
                , typename std::enable_if<
                           Kokkos::is_view< ViewType >::value &&
                           !Kokkos::is_reducer_type< ReducerType >::value
                  ,void*>::type = NULL)
    : m_functor( arg_functor )
    , m_policy( arg_policy )
    , m_reducer( InvalidType() )
    , m_result_ptr( arg_view.data() )
    { }

  inline
  ParallelReduce( const FunctorType & arg_functor
                , Policy       arg_policy
                , const ReducerType& reducer )
    : m_functor( arg_functor )
    , m_policy( arg_policy )
    , m_reducer( reducer )
    , m_result_ptr( reducer.view().data() )
    { }
};

} // namespace Impl
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

template< class FunctorType , class ... Traits >
class ParallelScan< FunctorType
                  , Kokkos::RangePolicy< Traits ... >
                  , Kokkos::Qthreads
                  >
{
private:

  typedef Kokkos::RangePolicy< Traits ... >  Policy ;

  typedef FunctorAnalysis< FunctorPatternInterface::SCAN , Policy , FunctorType > Analysis ;

  typedef typename Policy::work_tag     WorkTag ;
  typedef typename Policy::WorkRange    WorkRange ;
  typedef typename Policy::member_type  Member ;

  typedef Kokkos::Impl::FunctorValueInit<   FunctorType, WorkTag > ValueInit ;
  typedef Kokkos::Impl::FunctorValueJoin<   FunctorType, WorkTag > ValueJoin ;
  typedef Kokkos::Impl::FunctorValueOps<    FunctorType, WorkTag > ValueOps ;

  typedef typename Analysis::pointer_type    pointer_type ;
  typedef typename Analysis::reference_type  reference_type ;

        QthreadsExec * m_instance ;
  const FunctorType    m_functor ;
  const Policy         m_policy ;

  template< class TagType >
  inline static
  typename std::enable_if< std::is_same< TagType , void >::value >::type
  exec_range( const FunctorType & functor
            , const Member ibeg , const Member iend
            , reference_type update , const bool final )
    {
      for ( Member iwork = ibeg ; iwork < iend ; ++iwork ) {
        functor( iwork , update , final );
      }
    }

  template< class TagType >
  inline static
  typename std::enable_if< ! std::is_same< TagType , void >::value >::type
  exec_range( const FunctorType & functor
            , const Member ibeg , const Member iend
            , reference_type update , const bool final )
    {
      const TagType t{} ;
      for ( Member i = ibeg ; i < iend ; ++i ) {
        functor( t , i , update , final );
      }
    }

public:

  inline
  void execute() const
    {
      /*
      QthreadsExec::resize_worker_scratch( ValueTraits::value_size( m_functor ) , 0 );
      Impl::QthreadsExec::exec_all( Qthreads::instance() , & ParallelScan::exec , this );
      */
    }

  inline
  ParallelScan( const FunctorType & arg_functor
              , const Policy      & arg_policy
              )
    : m_instance ( t_qthreads_instance )
    , m_functor( arg_functor )
    , m_policy( arg_policy )
    {
    }
};

} // namespace Impl
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

template< class FunctorType , class ... Properties >
class ParallelFor< FunctorType
                 , Kokkos::TeamPolicy< Properties ... >
                 , Kokkos::Qthreads
                 >
{
private:

  enum { TEAM_REDUCE_SIZE = 512 };

  typedef Kokkos::Impl::TeamPolicyInternal< Kokkos::Qthreads, Properties ... > Policy ;
  typedef typename Policy::work_tag             WorkTag ;
  typedef typename Policy::schedule_type::type  SchedTag ;
  typedef typename Policy::member_type          Member ;

        QthreadsExec   * m_instance;
  const FunctorType    m_functor;
  const Policy         m_policy;
  const int            m_shmem_size;

  template< class TagType >
  inline static
  typename std::enable_if< ( std::is_same< TagType , void >::value ) >::type
  exec_team( const FunctorType & functor
           , HostThreadTeamData & data
           , const int league_rank_begin
           , const int league_rank_end
           , const int league_size )
    {
      std::cout << "exec_team 1 " << std::endl;
      for ( int r = league_rank_begin ; r < league_rank_end ; ) {

        functor( Member( data, r , league_size ) );

        if ( ++r < league_rank_end ) {
          // Don't allow team members to lap one another
          // so that they don't overwrite shared memory.
          if ( data.team_rendezvous() ) { data.team_rendezvous_release(); }
        }
      }
    }


  template< class TagType >
  inline static
  typename std::enable_if< ( ! std::is_same< TagType , void >::value ) >::type
  exec_team( const FunctorType & functor
           , HostThreadTeamData & data
           , const int league_rank_begin
           , const int league_rank_end
           , const int league_size )
    {
      const TagType t{};

      std::cout << " exec_team 2" << std::endl;
      for ( int r = league_rank_begin ; r < league_rank_end ; ) {

        functor( t , Member( data, r , league_size ) );

        if ( ++r < league_rank_end ) {
          // Don't allow team members to lap one another
          // so that they don't overwrite shared memory.
          if ( data.team_rendezvous() ) { data.team_rendezvous_release(); }
        }
      }
    }

public:

  inline
  void execute() const
    {
      std::cout << " execute 1 " << std::endl;
      enum { is_dynamic = std::is_same< SchedTag , Kokkos::Dynamic >::value };

      QthreadsExec::verify_is_master("Kokkos::Qthreads parallel_for");

      const size_t pool_reduce_size = 0 ; // Never shrinks
      const size_t team_reduce_size = TEAM_REDUCE_SIZE * m_policy.team_size();
      const size_t team_shared_size = m_shmem_size + m_policy.scratch_size(1);
      const size_t thread_local_size = 0 ; // Never shrinks

      m_instance->resize_thread_data( pool_reduce_size
                                    , team_reduce_size
                                    , team_shared_size
                                    , thread_local_size );

      const int pool_size = Qthreads::thread_pool_size();
      //#pragma omp parallel num_threads(pool_size)
      {
        HostThreadTeamData & data = *(m_instance->get_thread_data());

        const int active = data.organize_team( m_policy.team_size() );

        if ( active ) {
          data.set_work_partition( m_policy.league_size()
                                 , ( 0 < m_policy.chunk_size()
                                   ? m_policy.chunk_size()
                                   : m_policy.team_iter() ) );
        }

        if ( is_dynamic ) {
          // Must synchronize to make sure each team has set its
          // partition before begining the work stealing loop.
          if ( data.pool_rendezvous() ) data.pool_rendezvous_release();
        }

        if ( active ) {

          std::pair<int64_t,int64_t> range(0,0);

          do {

            range = is_dynamic ? data.get_work_stealing_chunk()
                               : data.get_work_partition();

            ParallelFor::template exec_team< WorkTag >
              ( m_functor , data
              , range.first , range.second , m_policy.league_size() );

          } while ( is_dynamic && 0 <= range.first );
        }

        data.disband_team();
      }
    }


  inline
  ParallelFor( const FunctorType & arg_functor ,
               const Policy      & arg_policy )
    : m_instance( t_qthreads_instance )
    , m_functor( arg_functor )
    , m_policy(  arg_policy )
    , m_shmem_size( arg_policy.scratch_size(0) +
                    arg_policy.scratch_size(1) +
                    FunctorTeamShmemSize< FunctorType >
                      ::value( arg_functor , arg_policy.team_size() ) )
    { std::cout <<  "parallel for with team shmem" << std::endl; }
};

//----------------------------------------------------------------------------

template< class FunctorType , class ReducerType, class ... Properties >
class ParallelReduce< FunctorType
                    , Kokkos::TeamPolicy< Properties ... >
                    , ReducerType
                    , Kokkos::Qthreads
                    >
{
private:

  enum { TEAM_REDUCE_SIZE = 512 };

  typedef Kokkos::Impl::TeamPolicyInternal< Kokkos::Qthreads, Properties ... >         Policy ;

  typedef FunctorAnalysis< FunctorPatternInterface::REDUCE , Policy , FunctorType > Analysis ;

  typedef typename Policy::work_tag             WorkTag ;
  typedef typename Policy::schedule_type::type  SchedTag ;
  typedef typename Policy::member_type          Member ;

  typedef Kokkos::Impl::if_c< std::is_same<InvalidType,ReducerType>::value
                            , FunctorType, ReducerType> ReducerConditional;

  typedef typename ReducerConditional::type ReducerTypeFwd;

  typedef Kokkos::Impl::FunctorValueInit<   ReducerTypeFwd , WorkTag >  ValueInit ;
  typedef Kokkos::Impl::FunctorValueJoin<   ReducerTypeFwd , WorkTag >  ValueJoin ;

  typedef typename Analysis::pointer_type    pointer_type ;
  typedef typename Analysis::reference_type  reference_type ;

        QthreadsExec   * m_instance;
  const FunctorType    m_functor;
  const Policy         m_policy;
  const ReducerType    m_reducer;
  const pointer_type   m_result_ptr;
  const int            m_shmem_size;

  template< class TagType >
  inline static
  typename std::enable_if< ( std::is_same< TagType , void >::value ) >::type
  exec_team( const FunctorType & functor
           , HostThreadTeamData & data
           , reference_type     & update
           , const int league_rank_begin
           , const int league_rank_end
           , const int league_size )
    {
      for ( int r = league_rank_begin ; r < league_rank_end ; ) {

        functor( Member( data, r , league_size ) , update );

        if ( ++r < league_rank_end ) {
          // Don't allow team members to lap one another
          // so that they don't overwrite shared memory.
          if ( data.team_rendezvous() ) { data.team_rendezvous_release(); }
        }
      }
    }


  template< class TagType >
  inline static
  typename std::enable_if< ( ! std::is_same< TagType , void >::value ) >::type
  exec_team( const FunctorType & functor
           , HostThreadTeamData & data
           , reference_type     & update
           , const int league_rank_begin
           , const int league_rank_end
           , const int league_size )
    {
      const TagType t{};

      for ( int r = league_rank_begin ; r < league_rank_end ; ) {

        functor( t , Member( data, r , league_size ) , update );

        if ( ++r < league_rank_end ) {
          // Don't allow team members to lap one another
          // so that they don't overwrite shared memory.
          if ( data.team_rendezvous() ) { data.team_rendezvous_release(); }
        }
      }
    }

public:

  inline
  void execute() const
    {
      enum { is_dynamic = std::is_same< SchedTag , Kokkos::Dynamic >::value };

      QthreadsExec::verify_is_master("Kokkos::Qthreads parallel_reduce");

      const size_t pool_reduce_size =
        Analysis::value_size( ReducerConditional::select(m_functor, m_reducer));

      const size_t team_reduce_size = TEAM_REDUCE_SIZE * m_policy.team_size();
      const size_t team_shared_size = m_shmem_size + m_policy.scratch_size(1);
      const size_t thread_local_size = 0 ; // Never shrinks

      m_instance->resize_thread_data( pool_reduce_size
                                    , team_reduce_size
                                    , team_shared_size
                                    , thread_local_size );

      const int pool_size = Qthreads::thread_pool_size();
      #pragma omp parallel num_threads(pool_size)
      {
        HostThreadTeamData & data = *(m_instance->get_thread_data());

        const int active = data.organize_team( m_policy.team_size() );

        if ( active ) {
          data.set_work_partition( m_policy.league_size()
                                 , ( 0 < m_policy.chunk_size()
                                   ? m_policy.chunk_size()
                                   : m_policy.team_iter() ) );
        }

        if ( is_dynamic ) {
          // Must synchronize to make sure each team has set its
          // partition before begining the work stealing loop.
          if ( data.pool_rendezvous() ) data.pool_rendezvous_release();
        }

        if ( active ) {
          reference_type update =
            ValueInit::init( ReducerConditional::select(m_functor , m_reducer)
                           , data.pool_reduce_local() );

          std::pair<int64_t,int64_t> range(0,0);

          do {

            range = is_dynamic ? data.get_work_stealing_chunk()
                               : data.get_work_partition();

            ParallelReduce::template exec_team< WorkTag >
              ( m_functor , data , update
              , range.first , range.second , m_policy.league_size() );

          } while ( is_dynamic && 0 <= range.first );
        } else {
          ValueInit::init( ReducerConditional::select(m_functor , m_reducer)
                           , data.pool_reduce_local() );
        }

        data.disband_team();

        //  This thread has updated 'pool_reduce_local()' with its
        //  contributions to the reduction.  The parallel region is
        //  about to terminate and the master thread will load and
        //  reduce each 'pool_reduce_local()' contribution.
        //  Must 'memory_fence()' to guarantee that storing the update to
        //  'pool_reduce_local()' will complete before this thread
        //  exits the parallel region.

        memory_fence();
      }

      // Reduction:

      const pointer_type ptr = pointer_type( m_instance->get_thread_data(0)->pool_reduce_local() );

      for ( int i = 1 ; i < pool_size ; ++i ) {
        ValueJoin::join( ReducerConditional::select(m_functor , m_reducer)
                       , ptr
                       , m_instance->get_thread_data(i)->pool_reduce_local() );
      }

      Kokkos::Impl::FunctorFinal<  ReducerTypeFwd , WorkTag >::final( ReducerConditional::select(m_functor , m_reducer) , ptr );

      if ( m_result_ptr ) {
        const int n = Analysis::value_count( ReducerConditional::select(m_functor , m_reducer) );

        for ( int j = 0 ; j < n ; ++j ) { m_result_ptr[j] = ptr[j] ; }
      }
    }

  //----------------------------------------

  template< class ViewType >
  inline
  ParallelReduce( const FunctorType  & arg_functor ,
                  const Policy       & arg_policy ,
                  const ViewType     & arg_result ,
                  typename std::enable_if<
                    Kokkos::is_view< ViewType >::value &&
                    !Kokkos::is_reducer_type<ReducerType>::value
                    ,void*>::type = NULL)
    : m_instance( t_qthreads_instance )
    , m_functor( arg_functor )
    , m_policy(  arg_policy )
    , m_reducer( InvalidType() )
    , m_result_ptr( arg_result.ptr_on_device() )
    , m_shmem_size( arg_policy.scratch_size(0) +
                    arg_policy.scratch_size(1) +
                    FunctorTeamShmemSize< FunctorType >
                      ::value( arg_functor , arg_policy.team_size() ) )
    {}

  inline
  ParallelReduce( const FunctorType & arg_functor
    , Policy       arg_policy
    , const ReducerType& reducer )
  : m_instance( t_qthreads_instance )
  , m_functor( arg_functor )
  , m_policy(  arg_policy )
  , m_reducer( reducer )
  , m_result_ptr(  reducer.view().data() )
  , m_shmem_size( arg_policy.scratch_size(0) +
                  arg_policy.scratch_size(1) +
                  FunctorTeamShmemSize< FunctorType >
                    ::value( arg_functor , arg_policy.team_size() ) )
  {
  /*static_assert( std::is_same< typename ViewType::memory_space
                          , Kokkos::HostSpace >::value
  , "Reduction result on Kokkos::OpenMP must be a Kokkos::View in HostSpace" );*/
  }

};

} // namespace Impl
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------



#endif
#endif /* #define KOKKOS_QTHREADS_PARALLEL_HPP */

