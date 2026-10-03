#![deny(missing_docs)]

//! This crate provides a mechanism to facilitate waiting for a type to change its state in an
//! asynchronous manner. It provides an extension for the users to implement the way the state would
//! change for a type via the [`State`] trait.
//!
//! # Usage
//!
//! ```
//! use async_flag::{Core, State};
//! use std::sync::atomic::{AtomicBool, Ordering};
//! use std::sync::Arc;
//!
//! struct Flag(AtomicBool);
//!
//! impl State for Flag {
//!     fn try_change(&self) {
//!         self.0.store(true, Ordering::SeqCst);
//!     }
//!
//!     fn has_changed(&self) -> bool {
//!         self.0.load(Ordering::SeqCst)
//!     }
//! }
//!
//! #[tokio::main]
//! async fn main() {
//!     let core = Arc::new(Core::<_, 8>::new(Flag(AtomicBool::new(false))));
//!     for _ in 0..8 {
//!         let core = Arc::clone(&core);
//!         tokio::spawn(async move {
//!             let wait_fut = core.try_wait();
//!             if let Some(f) = wait_fut {
//!                 println!("{:?}", f.await);
//!             }
//!         });
//!     }
//!
//!     tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
//!     core.try_wake_all();
//! }
//! ```

use arrayvec::ArrayVec;
use parking_lot::Mutex;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::pin::Pin;
use std::task::Waker;
use std::task::{Context, Poll};

const MAX_WAITERS: usize = 64;

/// A trait meant to be implemented for types that will specify when the wakers would be woken up.
pub trait State {
    /// Tries to change the state. Eg: Changing an `AtomicBool` from false to true. Once the state
    /// has changed, it cannot be reset.
    fn try_change(&self);

    /// Used to know whether the state has changed. If the state was changed on calling
    /// `try_change`, this must return true;
    fn has_changed(&self) -> bool;
}

/// The data structure that stores the type that implements the [`State`] trait along with the
/// `Waker`s meant to be woken up when the state changes.
///
/// Core is `Send` or `Sync` subject to `T` being `Send` or `Sync`.
pub struct Core<T, const N: usize> {
    core: T,
    waiters: Mutex<Waiters<N>>,
}

struct Waiters<const N: usize> {
    wakers: [MaybeUninit<Waker>; N],
    indices: ArrayVec<usize, N>,
    registered: u64,
    woken: bool,
}

macro_rules! big_guard {
    () => {
        struct Guard<'a, const N: usize> {
            waiters: &'a mut Waiters<N>,
        }

        impl<const N: usize> Drop for Guard<'_, N> {
            fn drop(&mut self) {
                if ::std::thread::panicking() {
                    // Drop all other wakers here. Rust initiates process abort if a thread panics
                    // while panicking. Therefore a panic while dropping other wakers is fine.
                    for (idx, element) in self.waiters.wakers.iter_mut().enumerate() {
                        if self.waiters.registered >> idx & 0b1 == 1 {
                            self.waiters.registered &= !(1 << idx);
                            unsafe {
                                element.assume_init_read().wake_by_ref();
                            }
                        }
                    }
                }
            }
        }
    };
}

// We need not reset here, because this being dropped means that the Core has been dropped.
impl<const N: usize> Drop for Waiters<N> {
    fn drop(&mut self) {
        big_guard!();
        let guard = Guard { waiters: self };

        for (idx, element) in guard.waiters.wakers.iter_mut().enumerate() {
            if guard.waiters.registered >> idx & 0b1 == 1 {
                guard.waiters.registered &= !(1 << idx);
                unsafe {
                    element.assume_init_read().wake_by_ref();
                }
            }
        }
    }
}

impl<const N: usize> Waiters<N> {
    fn new() -> Self {
        assert!(N <= MAX_WAITERS, "Max allowed waiters is {MAX_WAITERS}.");
        Self {
            wakers: unsafe { MaybeUninit::uninit().assume_init() },
            indices: ArrayVec::from(std::array::from_fn(|i| i)),
            woken: false,
            registered: 0,
        }
    }

    fn find_slot(&mut self) -> Option<usize> {
        self.indices.pop()
    }

    fn recover_idx(&mut self, idx: usize) {
        // We never push beyond N elements.
        self.indices.push(idx);
    }
}

impl<T, const N: usize> Core<T, N>
where
    T: State,
{
    /// Creates a new instance of [`Core`].
    pub fn new(core: T) -> Self {
        Self {
            core,
            waiters: Mutex::new(Waiters::new()),
        }
    }

    /// Attempts to get a slot in the collection of `Waker`s waiting for wakeup.
    pub fn try_wait(&self) -> Option<WaitFut<'_, T, N>> {
        if let Some(mut l) = self.waiters.try_lock()
            && let Some(idx) = l.find_slot()
        {
            Some(WaitFut {
                core: self,
                idx,
                _marker: PhantomData,
            })
        } else {
            None
        }
    }

    /// Attempts to change the state and if it succeeds, wakes up all the `Waker`s.
    pub fn try_wake_all(&self) {
        self.core.try_change();
        if self.core.has_changed() {
            let mut lock = self.waiters.lock();
            // Set woken to true so that further registration is not allowed.
            if lock.woken {
                return;
            } else {
                lock.woken = true;
            }
            big_guard!();
            let guard = Guard {
                waiters: &mut *lock,
            };

            for (idx, element) in guard.waiters.wakers.iter_mut().enumerate() {
                if guard.waiters.registered >> idx & 0b1 == 1 {
                    // Unset the registration bit so that we do not double drop the waker when the
                    // panic guard is dropped while panicking.
                    guard.waiters.registered &= !(1 << idx);
                    unsafe {
                        element.assume_init_read().wake_by_ref();
                    }
                }
            }
        }
    }
}

/// Future returned if [`Core::try_wait`] if a slot is obtained in the collection.
///
/// WaitFut is `Send` but not `Sync`.
pub struct WaitFut<'a, T, const N: usize> {
    core: &'a Core<T, N>,
    idx: usize,
    _marker: PhantomData<std::cell::Cell<()>>,
}

impl<T, const N: usize> WaitFut<'_, T, N> {
    /// Returns false if wakeup has already happened. Returns true otherwise.
    fn change_waker(&mut self, waker: &Waker) -> bool {
        struct Guard<'a, T, const N: usize> {
            core: &'a Core<T, N>,
            idx: usize,
        }

        impl<T, const N: usize> Drop for Guard<'_, T, N> {
            fn drop(&mut self) {
                if ::std::thread::panicking() {
                    // Set registered to false so that when Waitfut gets dropped we do not drop the
                    // waker again.
                    self.core.waiters.lock().registered &= !(1 << self.idx);
                }
            }
        }

        let guard = Guard {
            core: self.core,
            idx: self.idx,
        };

        let mut lock = guard.core.waiters.lock();

        if lock.woken {
            return false;
        }

        if lock.registered >> self.idx & 0b1 == 1 {
            unsafe {
                if !lock.wakers[self.idx].assume_init_ref().will_wake(waker) {
                    lock.wakers[self.idx].assume_init_drop();
                } else {
                    return true;
                }
            }
        }
        lock.wakers[self.idx] = MaybeUninit::new(waker.clone());
        lock.registered |= 1 << self.idx;
        true
    }
}

impl<T, const N: usize> Drop for WaitFut<'_, T, N> {
    fn drop(&mut self) {
        let mut lock = self.core.waiters.lock();
        lock.recover_idx(self.idx);
        if lock.registered >> self.idx & 0b1 == 1 {
            lock.registered &= !(1 << self.idx);
            unsafe {
                lock.wakers[self.idx].assume_init_drop();
            }
        }
    }
}

impl<T, const N: usize> Future for WaitFut<'_, T, N>
where
    T: State,
{
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.core.core.has_changed() || !self.change_waker(cx.waker()) {
            Poll::Ready(())
        } else {
            Poll::Pending
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(miri))]
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::task::Wake;
    use std::time::Duration;

    impl State for AtomicBool {
        fn try_change(&self) {
            self.store(true, Ordering::SeqCst);
        }

        fn has_changed(&self) -> bool {
            self.load(Ordering::SeqCst)
        }
    }

    #[tokio::test]
    async fn test_concurrent() {
        let core = Arc::new(Core::<_, 8>::new(AtomicBool::new(false)));
        for _ in 0..7 {
            let core = core.clone();
            tokio::spawn(async move {
                let wait_fut = core.try_wait();
                if let Some(f) = wait_fut {
                    println!("{:?}", f.await);
                }
            });
        }

        tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;
        core.try_wake_all();
    }

    #[test]
    fn check_wake() {
        let core = Core::<_, 2>::new(AtomicBool::new(false));
        core.try_wake_all();
    }

    // Code beyond this point have been generated with Claude.

    struct Counter {
        wakes: AtomicUsize,
        explode: bool,
    }

    impl Wake for Counter {
        fn wake(self: Arc<Self>) {
            Wake::wake_by_ref(&self);
        }

        fn wake_by_ref(self: &Arc<Self>) {
            self.wakes.fetch_add(1, Ordering::SeqCst);
            if self.explode {
                panic!("boom");
            }
        }
    }

    fn counter_waker(explode: bool) -> (Arc<Counter>, Waker) {
        let c = Arc::new(Counter {
            wakes: AtomicUsize::new(0),
            explode,
        });
        let w = Waker::from(Arc::clone(&c));
        (c, w)
    }

    fn wakes(c: &Arc<Counter>) -> usize {
        c.wakes.load(Ordering::SeqCst)
    }

    #[test]
    fn pending_until_woken_then_ready() {
        let core = Core::<_, 2>::new(AtomicBool::new(false));
        let mut fut = core.try_wait().unwrap();
        let (c, w) = counter_waker(false);
        let mut cx = Context::from_waker(&w);

        assert!(Pin::new(&mut fut).poll(&mut cx).is_pending());
        assert_eq!(wakes(&c), 0);
        assert_eq!(Arc::strong_count(&c), 3);

        core.try_wake_all();
        assert_eq!(wakes(&c), 1);
        assert_eq!(Arc::strong_count(&c), 2);
        assert!(Pin::new(&mut fut).poll(&mut cx).is_ready());
    }

    #[test]
    fn unpolled_future_survives_wake() {
        let core = Core::<_, 2>::new(AtomicBool::new(false));
        let mut fut = core.try_wait().unwrap();
        core.try_wake_all();

        let (c, w) = counter_waker(false);
        let mut cx = Context::from_waker(&w);
        assert!(Pin::new(&mut fut).poll(&mut cx).is_ready());
        assert_eq!(wakes(&c), 0);
        assert_eq!(Arc::strong_count(&c), 2);
    }

    #[test]
    fn late_waiter_is_ready_immediately() {
        let core = Core::<_, 2>::new(AtomicBool::new(false));
        core.try_wake_all();

        let mut fut = core.try_wait().unwrap();
        let (c, w) = counter_waker(false);
        let mut cx = Context::from_waker(&w);
        assert!(Pin::new(&mut fut).poll(&mut cx).is_ready());
        assert_eq!(Arc::strong_count(&c), 2);
    }

    #[test]
    fn second_wake_is_a_noop() {
        let core = Core::<_, 2>::new(AtomicBool::new(false));
        let mut fut = core.try_wait().unwrap();
        let (c, w) = counter_waker(false);
        let mut cx = Context::from_waker(&w);
        assert!(Pin::new(&mut fut).poll(&mut cx).is_pending());

        core.try_wake_all();
        core.try_wake_all();
        assert_eq!(wakes(&c), 1);
    }

    #[test]
    fn drop_before_wake_releases_waker_and_slot() {
        let core = Core::<_, 1>::new(AtomicBool::new(false));
        let (c, w) = counter_waker(false);
        let mut cx = Context::from_waker(&w);

        let mut fut = core.try_wait().unwrap();
        assert!(core.try_wait().is_none());
        assert!(Pin::new(&mut fut).poll(&mut cx).is_pending());
        assert_eq!(Arc::strong_count(&c), 3);

        drop(fut);
        assert_eq!(Arc::strong_count(&c), 2);
        assert!(core.try_wait().is_some());

        core.try_wake_all();
        assert_eq!(wakes(&c), 0);
    }

    #[test]
    fn capacity_is_enforced_and_slots_recycle() {
        let core = Core::<_, 2>::new(AtomicBool::new(false));
        let a = core.try_wait().unwrap();
        let _b = core.try_wait().unwrap();
        assert!(core.try_wait().is_none());

        drop(a);
        let _c = core.try_wait().unwrap();
        assert!(core.try_wait().is_none());
    }

    // This test reports a false positive when run with MIRI in parking_lot_core.
    #[cfg(not(miri))]
    #[test]
    fn panicking_wake_does_not_leak_wakers() {
        let core = Core::<_, 4>::new(AtomicBool::new(false));
        let mut futs = Vec::new();
        let mut handles = Vec::new();
        for i in 0..3 {
            let (c, w) = counter_waker(i == 1);
            let mut fut = core.try_wait().unwrap();
            assert!(
                Pin::new(&mut fut)
                    .poll(&mut Context::from_waker(&w))
                    .is_pending()
            );
            futs.push(fut);
            handles.push((c, w));
        }

        let r = catch_unwind(AssertUnwindSafe(|| core.try_wake_all()));
        assert!(r.is_err());

        for (c, _w) in &handles {
            assert_eq!(wakes(c), 1, "every waker must still get woken");
            assert_eq!(
                Arc::strong_count(c),
                2,
                "every stored clone must be dropped"
            );
        }

        for mut f in futs {
            let (_c, w) = counter_waker(false);
            assert!(
                Pin::new(&mut f)
                    .poll(&mut Context::from_waker(&w))
                    .is_ready()
            );
        }
    }

    #[test]
    fn repoll_with_same_waker_does_not_clone_again() {
        let core = Core::<_, 1>::new(AtomicBool::new(false));
        let mut fut = core.try_wait().unwrap();
        let (c, w) = counter_waker(false);
        let mut cx = Context::from_waker(&w);

        for _ in 0..3 {
            assert!(Pin::new(&mut fut).poll(&mut cx).is_pending());
        }
        assert_eq!(Arc::strong_count(&c), 3);
    }

    #[test]
    fn repoll_with_new_waker_replaces_old_one() {
        let core = Core::<_, 1>::new(AtomicBool::new(false));
        let mut fut = core.try_wait().unwrap();
        let (c1, w1) = counter_waker(false);
        let (c2, w2) = counter_waker(false);

        assert!(
            Pin::new(&mut fut)
                .poll(&mut Context::from_waker(&w1))
                .is_pending()
        );
        assert!(
            Pin::new(&mut fut)
                .poll(&mut Context::from_waker(&w2))
                .is_pending()
        );
        assert_eq!(Arc::strong_count(&c1), 2);
        assert_eq!(Arc::strong_count(&c2), 3);

        core.try_wake_all();
        assert_eq!(wakes(&c1), 0);
        assert_eq!(wakes(&c2), 1);
    }

    #[test]
    fn leaked_future_is_cleaned_up_by_core_drop() {
        let (c, w) = counter_waker(false);
        {
            let core = Core::<_, 2>::new(AtomicBool::new(false));
            let mut fut = core.try_wait().unwrap();
            assert!(
                Pin::new(&mut fut)
                    .poll(&mut Context::from_waker(&w))
                    .is_pending()
            );
            std::mem::forget(fut);
            assert_eq!(Arc::strong_count(&c), 3);
        }
        assert_eq!(wakes(&c), 1);
        assert_eq!(Arc::strong_count(&c), 2);
    }

    #[test]
    fn leaked_unpolled_future_is_safe_for_wake_and_drop() {
        let core = Core::<_, 2>::new(AtomicBool::new(false));
        std::mem::forget(core.try_wait().unwrap());
        core.try_wake_all();
        drop(core);
    }

    #[test]
    fn no_lost_wakeup_race() {
        let iters = if cfg!(miri) { 20 } else { 5_000 };
        for _ in 0..iters {
            let core = Core::<_, 1>::new(AtomicBool::new(false));
            let mut fut = core.try_wait().unwrap();
            let (c, w) = counter_waker(false);
            let w_ref = &w;
            let fut_ref = &mut fut;

            let ready = std::thread::scope(|s| {
                let poller = s.spawn(move || {
                    let mut cx = Context::from_waker(w_ref);
                    Pin::new(fut_ref).poll(&mut cx).is_ready()
                });
                s.spawn(|| core.try_wake_all());
                poller.join().unwrap()
            });

            assert!(ready || wakes(&c) == 1, "lost wakeup");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn all_waiters_complete() {
        let core = Arc::new(Core::<_, 8>::new(AtomicBool::new(false)));
        let mut handles = Vec::new();
        for _ in 0..8 {
            let core = Arc::clone(&core);
            handles.push(tokio::spawn(async move {
                let fut = loop {
                    if let Some(f) = core.try_wait() {
                        break f;
                    }
                    tokio::task::yield_now().await;
                };
                fut.await;
            }));
        }

        tokio::time::sleep(Duration::from_millis(50)).await;
        core.try_wake_all();

        for h in handles {
            tokio::time::timeout(Duration::from_secs(5), h)
                .await
                .expect("waiter never woke")
                .unwrap();
        }
    }
}
