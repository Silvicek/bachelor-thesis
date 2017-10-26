Neural networks for humanoid robot control
==========================================

Pendulum:
---------

For testing DDPG on the pendulum task, go to py/ subfolder and run

    python [pendulum.py/pendulum_rnn.py] --id [your id] --task [swingup/balance]
    
NOTE: the run doesn't stop on its own  
to visualize the best found policy, run

    python [pendulum.py/pendulum_rnn.py] --id [your id] --task [swingup/balance] -v
    
this will create a video in py/pendulum/[task]/[your id]/video

you can play with the networks and training settings in the pendulu.py/pendulum_rnn.py scripts

Dependencies:
-------------
Python 2.7+scipy+cython+Theano

Robot:
------
Place dlbw folder to V-REP_PRO_EDU_V3_2_2_64_Linux/programming  
to build the project run

    ./dlbw.sh
    
note that this requires lua and Qt libraries in PATH and the py/ subfolder in your PYTHONPATH

to test the project, execute V-REP with options:

    ./vrep.sh programming/dlbw/scenes/NAO_noPID.ttt -ddpg -s -h -q

    
Dependencies:
-------------
Python 2.7+scipy+cython+Theano  
Qt 5.5  
Lua 5.1