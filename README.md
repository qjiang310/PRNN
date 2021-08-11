# PRNN
Building command: scons mode=release install=true install_path=./build_local

To resolve the bad_alloc error, do the following export commands:
  export LD_LIBRARY_PATH=/s/mpfr-3.1.6/amd64_ubu20/lib:/s/gcc-5.4.0/amd64_ubu20/lib64:/s/gcc-5.4.0/amd64_ubu20/lib:$LD_LIBRARY_PATH
  export PATH=/s/gcc-5.4.0/amd64_ubu20/bin:$PATH
  
Run the benchmarks:build_local/bin/persistent-rnn-benchmark -backend persistent --mini-batch-size # --layer-size # 
To resolve the error while loading shared libraries, export PRNN/build_local/lib
  
