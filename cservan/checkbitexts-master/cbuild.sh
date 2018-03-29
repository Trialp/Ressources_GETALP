
install -d build
pushd build
#cmake -DCMAKE_INSTALL_PREFIX=$HOME/Programmation/TEMP -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
v_uname=`uname`
/usr/bin/cmake -DCMAKE_INSTALL_PREFIX=~/Tools/checkbitexts/  -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
make -j4 && \
make install
popd
