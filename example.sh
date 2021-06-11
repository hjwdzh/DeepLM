#download submodule
git submodule update --init --recursive

#compile
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DWITH_CUDA=ON
make -j8
export PYTHONPATH=$PYTHONPATH:$(pwd)
cd ..

#prepare data
mkdir data
cd data
wget https://grail.cs.washington.edu/projects/bal/data/ladybug/problem-49-7776-pre.txt.bz2 --no-check-certificate
bzip2 -d problem-49-7776-pre.txt.bz2
cd ..

#global lm solver
TORCH_USE_RTLD_GLOBAL=YES python3 examples/BundleAdjuster/bundle_adjuster.py --balFile ./data/problem-49-7776-pre.txt --device cuda
