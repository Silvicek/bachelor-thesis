#!/bin/sh
cd py
python setup.py build_ext --inplace
cd ..
PATH="/home/silvicek/Qt/5.5/gcc_64/bin:$PATH"
QTDIR=/home/silvicek/Qt/5.5/gcc_64
export QTDIR PATH
qmake vrep.pro -r -spec linux-g++
make
mv *.o build
mv vrep build
cp build/vrep ../..

