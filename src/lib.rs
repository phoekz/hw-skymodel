//! # Hosek-Wilkie Skylight Model
//!
//! This crate contains a pure Rust implementation of [Hosek-Wilkie Skylight
//! Model](https://cgg.mff.cuni.cz/projects/SkylightModelling/). This
//! implementation is intended to be used in interactive computer graphics
//! applications, which is why we made the following changes compared to the
//! original ANSI C implementation:
//!
//! - Only RGB datasets and functions are included. Interactive applications
//!   don't typically deal with spectral forms and non-RGB color space, which is
//!   why they were excluded from this implementation.
//! - Arithmetic and storage precision is lowered from `f64` to `f32`. Lower
//!   precision has a minimal impact in visual quality, while improving storage
//!   and computational costs.
//! - Removed solar radiance and "alien-like worlds" functionality.
//!

/// RGB version of the model.
pub mod rgb;
