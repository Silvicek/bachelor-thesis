from ddpg.ddpg import DDPG, Holder, save_ddpg, load_ddpg
from cartpole import InvertedPendulum, animate_pendulum
import numpy as np
from math import sin, cos, pi
import theano

bestval = -9999.


def create_dirs():
    import os
    def create_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)
    dir = config['dir']+'/'+config['task']+'/'+config['id']
    create_dir(dir)
    create_dir(dir+'/models')
    create_dir(dir+'/plots')
    create_dir(dir+'/videos')
    print 'creating', dir


def save(e, t, score):
    global bestval
    print 'score, best', score, bestval, e, config['_f']
    if e % config['_f'] == 0 and score > bestval:
        path = config['dir']+'/'+config['task']+'/'+config['id']+'/models/bestval.model'
        save_ddpg(model, path)
        bestval = score
        print 'best model saved'
    if e % config['save_f'] == 0:
        path = config['dir']+'/'+config['task']+'/'+config['id']+'/models/episode_'+str(e)+'.model'
        save_ddpg(model, path)
        print str(e)+'. model saved'

def act(step):
    return True
    # return step % 3 == 0

def r_balance(s, a, pendulum):
    return pendulum.dt * (abs(s[0, 2]) - .1*abs(a[0,0]) - .3*abs(s[0,0]))

def r_swingup(s, a, pendulum):
    return pendulum.dt * (abs(s[0, 2]) - .1*abs(a[0,0]) - .1*abs(s[0,0]))

def end_balance(s, pendulum):
    return pendulum.t > pendulum.end or abs(s[0, 2]) < pi/2\
                or not (-10 < s[0,0] < 10)

def end_swingup(s, pendulum):
    return pendulum.t > pendulum.end or not (-10 < s[0,0] < 10)

def episode(pendulum, e):
    balance_mode = config['task'] == 'balance'
    r_func = r_balance if balance_mode else r_swingup
    end_func = end_balance if balance_mode else end_swingup

    score = 0.
    r = 0.
    run = []
    s_ = np.array([pendulum.get_state()]).astype(theano.config.floatX)
    while True:
        run.append([pendulum.t] + pendulum.get_state())
        if act(int(pendulum.t/pendulum.dt)):
            if e % config['_f'] == 0:
                a = model.fwp_actor_(s_)
                # a = model.stoch_action_(s_)
            else:
                a = model.stoch_action(s_)

        if e % config['save_f'] == 0:
            print 't, s_=', pendulum.t, s_
            print 'r, c=', r, model.fwp_critic_(s_, a)
            print 'a=', a

        pendulum.update(a[0,0])

        s = np.array([pendulum.get_state()]).astype(theano.config.floatX)

        r = r_func(s, a, pendulum)

        if end_func(s, pendulum):
            if pendulum.t < pendulum.end:
                r = np.array(-.1)
            holder = Holder(s_.flatten(), a.flatten(), False)
            holder.complete(r.astype(theano.config.floatX), s.flatten())
            model.R_add(holder)
            break
        if not act(int(pendulum.t/pendulum.dt)):
            continue

        holder = Holder(s_.flatten(), a.flatten())
        score += r
        holder.complete(r.astype(theano.config.floatX), s.flatten())
        model.R_add(holder)

        if e % config['_f'] == 0:
            pass
        else:
            model.train(True)

        if not balance_mode and pi-abs(s[0, 2]) < 0.1:
            r += .1
            # print 'round&round'

        if model.time_steps % 1000 == 0:
            print 'TIME AT', model.time_steps, '=', TIME.time()-start_time

        s_ = s

    save(e, pendulum.t, score)
    return score, pendulum.t

import time as TIME
start_time = TIME.time()
episodes = 100000
def train(pendulum, start):
    t_ = []
    r_ = []
    for e in range(episodes):
        score, time = episode(pendulum, e)

        if e % config['_f'] == 0:
            t_.append(model.time_steps)
            r_.append(score)
            path = config['dir']+'/'+config['task']+'/'+config['id']+'/plots/plot.npy'
            np.save(open(path, 'wb'), np.array([t_, r_]))

        print e, '...', 'score =', score, ', time =', time
        if e % config['_f'] == config['_f']-1:
            pendulum.set_state(np.array([0, 0., start, 0.], dtype=theano.config.floatX))
        else:
            pendulum.set_state(np.array([0, 0., np.random.normal(start, scale=0.1), 0.]).astype(theano.config.floatX))
        pendulum.t = 0.
        model.noise.reset()
        print 'max=', max([np.max(np.abs(par.eval())) for par in model.actor.nn.params])


def config_init():
    c = dict()
    c['task'] = 'balance'
    c['dir'] = 'pendulum'
    c['_f'] = 10
    c['save_f'] = 100
    c['v'] = False
    c['plot'] = None
    return c


def config_finalize(args):
    import random
    config['id'] = ''.join(random.choice('0123456789abcdef') for i in range(5))
    config['string'] = ''
    for key in args.keys():
        if args[key] is not None:
            config[key] = args[key]

import matplotlib.pyplot as plt
def plot_t_r(path):
    x = np.load(open(path))
    plt.plot(x[0], x[1])
    plt.show()


config = config_init()
import argparse
import sys
if __name__ == '__main__':
    # np.random.seed(13377)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir')
    parser.add_argument('--task')
    parser.add_argument('--path')
    parser.add_argument('--id')
    parser.add_argument('-v', action='store_true')
    parser.add_argument('--plot')
    args = parser.parse_args()

    config_finalize(vars(args))

    if config['plot'] is not None:
        plot_t_r(config['plot'])
        sys.exit()

    #                        critic|actor
    alpha, beta, gamma, tau = 1e-3, 1e-4, 0.99, 1e-3
    consts = alpha, beta, gamma, tau
    start = pi if config['task'] == 'balance' else 0.
    init_conds = np.array([0, 0., start, 0.], dtype=theano.config.floatX)
    create_dirs()

    if not config['v']:
        # ===================== change training parameters here
        model = DDPG(dt=1e-2,
                     batch_size=64,
                     hidden_a=[50,50],
                     hidden_c=[100,100],
                     consts=consts,
                     s_dim=4,
                     a_dim=1,
                     optimizers=['adam', 'adam'])

        # ===================== train
        pendulum = InvertedPendulum(init_conds[:2], init_conds[2:], end=20.)
        train(pendulum, start)
    else:
        # ===================== visualize
        path = config['dir']+'/'+config['task']+'/'+config['id']+'/models/bestval.model'
        if config.get('path'):
            path = config['path']
        model = load_ddpg(path)
        control = model.fwp_actor_
        # control = model.stoch_action_
        pendulum = InvertedPendulum(init_conds[:2], init_conds[2:], end=20.)
        data = pendulum.integrate(control)
        t = data[:,0]
        x = data[:,1]
        theta = data[:,3]
        xx = np.vstack((x,theta)).T
        path = config['dir']+'/'+config['task']+'/'+config['id']+'/videos/bestval.mp4'
        animate_pendulum(t, xx, filename=path)
