// An Analytic Model for Full Spectral Sky-Dome Radiance
// Lukas Hosek & Alexander Wilkie
// Project page: https://cgg.mff.cuni.cz/projects/SkylightModelling/
// License file: hosek-wilkie-license.txt

use std::f32::consts::{FRAC_PI_2, PI};

use embed_doc_image::embed_doc_image;
use thiserror::Error;

/// R, G, or B channel.
pub enum Channel {
    R = 0,
    G = 1,
    B = 2,
}

/// Initial parameters for the sky model.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct SkyParams {
    /// (Solar) elevation is given in radians. Elevation must be between `0..=π/2`.
    pub elevation: f32,
    /// Turbidity must be between `1..=10`.
    pub turbidity: f32,
    /// (Ground) albedo must be between `0..=1`.
    pub albedo: [f32; 3],
}

impl Default for SkyParams {
    fn default() -> Self {
        Self {
            elevation: 0.0,
            turbidity: 1.0,
            albedo: [1.0, 1.0, 1.0],
        }
    }
}

#[derive(Error, PartialEq, Debug)]
pub enum Error {
    #[error("Elevation must be between 0..=π/2, got {0} instead")]
    ElevationOutOfRange(f32),
    #[error("Turbidity must be between 1..=10, got {0} instead")]
    TurbidityOutOfRange(f32),
    #[error("Albedo must be between 0..=1, got {0:?} instead")]
    AlbedoOutOfRange([f32; 3]),
}

/// The state of the sky model. Ideally, you should keep the state around as
/// long as any of the `SkyParams` parameters don't change. If any parameters
/// change, you must re-create the `SkyState` object.
///
/// Note: if you are planning to run the model in a shader, you only need to
/// translate [`SkyState::radiance`] into shader code. The rest can be executed
/// on the CPU. You grab the raw parameters using [`SkyState::raw`] and upload
/// the 30 `f32`s to the GPU in a uniform buffer, for example.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct SkyState {
    params: [f32; 27],
    radiances: [f32; 3],
}

impl SkyState {
    /// Creates `SkyState`.
    ///
    /// # Errors
    /// Can fail if any of the parameters are out of range.
    pub fn new(sky_params: &SkyParams) -> Result<Self, Error> {
        // Validate parameters.
        if !(0.0..=FRAC_PI_2).contains(&sky_params.elevation) {
            return Err(Error::ElevationOutOfRange(sky_params.elevation));
        }
        if !(1.0..=10.0).contains(&sky_params.turbidity) {
            return Err(Error::TurbidityOutOfRange(sky_params.turbidity));
        }
        for albedo in sky_params.albedo {
            if !(0.0..=1.0).contains(&albedo) {
                return Err(Error::AlbedoOutOfRange(sky_params.albedo));
            }
        }

        // Load datasets.
        macro_rules! dataset {
            ($path: literal) => {{
                // Note: Rust's include_bytes! doesn't guarantee that the byte
                // slice is aligned. That's why we use include_bytes_aligned
                // crate to ensure 4-byte alignment so we can cast into
                // f32-slice.
                use include_bytes_aligned::include_bytes_aligned;
                let bytes = include_bytes_aligned!(4, $path);
                bytemuck::cast_slice::<_, f32>(bytes)
            }};
        }
        let params_r = dataset!("params-r");
        let params_g = dataset!("params-g");
        let params_b = dataset!("params-b");
        let radiances_r = dataset!("radiances-r");
        let radiances_g = dataset!("radiances-g");
        let radiances_b = dataset!("radiances-b");

        // Init state.
        let mut params = [0.0; 27];
        let mut radiances = [0.0; 3];
        let elevation = sky_params.elevation;
        let turbidity = sky_params.turbidity;
        let albedo = sky_params.albedo;
        let t = (elevation / (0.5 * PI)).powf(1.0 / 3.0);

        init_params(&mut params[..], params_r, turbidity, albedo[0], t);
        init_params(&mut params[9..], params_g, turbidity, albedo[1], t);
        init_params(&mut params[(9 * 2)..], params_b, turbidity, albedo[2], t);
        init_radiances(&mut radiances[0], radiances_r, turbidity, albedo[0], t);
        init_radiances(&mut radiances[1], radiances_g, turbidity, albedo[1], t);
        init_radiances(&mut radiances[2], radiances_b, turbidity, albedo[2], t);

        Ok(Self { params, radiances })
    }

    /// Evaluates incoming radiance for a given channel. See the figure on how
    /// `theta` and `gamma` are defined.
    ///
    /// ![Coordinate system of the sky model][coordinate-system]
    ///
    #[embed_doc_image("coordinate-system", "images/coordinate-system.png")]
    pub fn radiance(&self, theta: f32, gamma: f32, channel: Channel) -> f32 {
        let channel = channel as usize;
        let r = self.radiances[channel];
        let p = &self.params[(9 * channel)..];
        let p0 = p[0];
        let p1 = p[1];
        let p2 = p[2];
        let p3 = p[3];
        let p4 = p[4];
        let p5 = p[5];
        let p6 = p[6];
        let p7 = p[7];
        let p8 = p[8];

        let cos_gamma = gamma.cos();
        let cos_gamma2 = cos_gamma * cos_gamma;
        let cos_theta = theta.cos().abs();

        let exp_m = (p4 * gamma).exp();
        let ray_m = cos_gamma2;
        let mie_m_lhs = 1.0 + cos_gamma2;
        let mie_m_rhs = (1.0 + p8 * p8 - 2.0 * p8 * cos_gamma).powf(1.5);
        let mie_m = mie_m_lhs / mie_m_rhs;
        let zenith = cos_theta.sqrt();
        let radiance_lhs = 1.0 + p0 * (p1 / (cos_theta + 0.01)).exp();
        let radiance_rhs = p2 + p3 * exp_m + p5 * ray_m + p6 * mie_m + p7 * zenith;
        let radiance_dist = radiance_lhs * radiance_rhs;
        r * radiance_dist
    }

    /// Returns the internal state. Used when [`SkyState::radiance`] is
    /// implemented and executed externally, for example as a GPU shader.
    pub fn raw(&self) -> ([f32; 27], [f32; 3]) {
        (self.params, self.radiances)
    }
}

fn init_params(out_params: &mut [f32], dataset: &[f32], turbidity: f32, albedo: f32, t: f32) {
    let turbidity_int = turbidity.trunc() as usize;
    let turbidity_rem = turbidity.fract();
    let turbidity_min = turbidity_int.saturating_sub(1);
    let turbidity_max = turbidity_int.min(9);
    let p0 = &dataset[(9 * 6 * turbidity_min)..];
    let p1 = &dataset[(9 * 6 * turbidity_max)..];
    let p2 = &dataset[(9 * 6 * 10 + 9 * 6 * turbidity_min)..];
    let p3 = &dataset[(9 * 6 * 10 + 9 * 6 * turbidity_max)..];
    let s0 = (1.0 - albedo) * (1.0 - turbidity_rem);
    let s1 = (1.0 - albedo) * turbidity_rem;
    let s2 = albedo * (1.0 - turbidity_rem);
    let s3 = albedo * turbidity_rem;

    for i in 0..9 {
        out_params[i] += s0 * quintic::<9>(&p0[i..], t);
        out_params[i] += s1 * quintic::<9>(&p1[i..], t);
        out_params[i] += s2 * quintic::<9>(&p2[i..], t);
        out_params[i] += s3 * quintic::<9>(&p3[i..], t);
    }
}

fn init_radiances(out_radiance: &mut f32, dataset: &[f32], turbidity: f32, albedo: f32, t: f32) {
    let turbidity_int = turbidity.trunc() as usize;
    let turbidity_rem = turbidity.fract();
    let turbidity_min = turbidity_int.saturating_sub(1);
    let turbidity_max = turbidity_int.min(9);
    let p0 = &dataset[(6 * turbidity_min)..];
    let p1 = &dataset[(6 * turbidity_max)..];
    let p2 = &dataset[(6 * 10 + 6 * turbidity_min)..];
    let p3 = &dataset[(6 * 10 + 6 * turbidity_max)..];
    let s0 = (1.0 - albedo) * (1.0 - turbidity_rem);
    let s1 = (1.0 - albedo) * turbidity_rem;
    let s2 = albedo * (1.0 - turbidity_rem);
    let s3 = albedo * turbidity_rem;

    *out_radiance += s0 * quintic::<1>(p0, t);
    *out_radiance += s1 * quintic::<1>(p1, t);
    *out_radiance += s2 * quintic::<1>(p2, t);
    *out_radiance += s3 * quintic::<1>(p3, t);
}

fn quintic<const STRIDE: usize>(p: &[f32], t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t2 * t2;
    let t5 = t4 * t;

    let inv_t = 1.0 - t;
    let inv_t2 = inv_t * inv_t;
    let inv_t3 = inv_t2 * inv_t;
    let inv_t4 = inv_t2 * inv_t2;
    let inv_t5 = inv_t4 * inv_t;

    let m0 = p[0] * inv_t5;
    let m1 = p[STRIDE] * 5.0 * inv_t4 * t;
    let m2 = p[2 * STRIDE] * 10.0 * inv_t3 * t2;
    let m3 = p[3 * STRIDE] * 10.0 * inv_t2 * t3;
    let m4 = p[4 * STRIDE] * 5.0 * inv_t * t4;
    let m5 = p[5 * STRIDE] * t5;

    m0 + m1 + m2 + m3 + m4 + m5
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default() {
        let state = SkyState::new(&SkyParams::default());
        assert!(state.is_ok());
    }

    #[test]
    fn elevation_out_of_range() {
        let params = SkyParams {
            elevation: -1.0,
            ..SkyParams::default()
        };
        let state = SkyState::new(&params);
        assert_eq!(state, Err(Error::ElevationOutOfRange(-1.0)));
        assert_eq!(
            "Elevation must be between 0..=π/2, got -1 instead",
            format!("{}", state.unwrap_err())
        );
    }

    #[test]
    fn turbidity_out_of_range() {
        let params = SkyParams {
            turbidity: 0.0,
            ..SkyParams::default()
        };
        let state = SkyState::new(&params);
        assert_eq!(state, Err(Error::TurbidityOutOfRange(0.0)));
        assert_eq!(
            "Turbidity must be between 1..=10, got 0 instead",
            format!("{}", state.unwrap_err())
        );
    }

    #[test]
    fn albedo_out_of_range() {
        let params = SkyParams {
            albedo: [0.0, 2.0, 0.0],
            ..SkyParams::default()
        };
        let state = SkyState::new(&params);
        assert_eq!(state, Err(Error::AlbedoOutOfRange([0.0, 2.0, 0.0])));
        assert_eq!(
            "Albedo must be between 0..=1, got [0.0, 2.0, 0.0] instead",
            format!("{}", state.unwrap_err())
        );
    }
}
