mkdir -p repos
cd repos
echo `pwd`
git clone git@github.com:tomaszkacprzak/euclid-multiprobe-deeplss-training.git
git clone git@github.com:tomaszkacprzak/euclid-cosmology-multiprobe-photometric-sbi.git
git clone git@github.com:tomaszkacprzak/euclid-multiprobe-simulation-forward-model.git
git clone git@github.com:tomaszkacprzak/euclid-cosmogridv11.git
git clone git@github.com:tomaszkacprzak/deepsphere-cosmo-pytorch.git
git clone git@github.com:tomaszkacprzak/FAST-PT.git
git clone git@github.com:tomaszkacprzak/euclid-multiprobe-simulation-inference.git
git clone git@github.com:tomaszkacprzak/CCL.git
git clone --branch cosmogrid_des_y3 --single-branch https://cosmo-gitlab.phys.ethz.ch/cosmo/UFalcon.git
cd ..
echo `pwd`