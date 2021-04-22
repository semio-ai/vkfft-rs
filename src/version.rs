use std::fmt::{Display, Formatter};
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Version {
  major: u32,
  minor: u32,
  patch: u32,
}

impl Version {
  #[inline]
  pub fn major(&self) -> u32 {
    self.major
  }

  #[inline]
  pub fn minor(&self) -> u32 {
    self.minor
  }

  #[inline]
  pub fn patch(&self) -> u32 {
    self.patch
  }
}

impl Display for Version {
  fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
    write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
  }
}

pub fn version() -> Version {
  let ver = unsafe { vkfft_sys::VkFFTGetVersion() };

  Version {
    major: ver / 10000,
    minor: ver % 10000 / 100,
    patch: ver % 100,
  }
}
