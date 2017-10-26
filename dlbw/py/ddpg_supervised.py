
import numpy as np
import theano
import matplotlib.pyplot as plt
from ddpg.ddpg import DDPG, save_ddpg


def keras_model(steps, hidden):
    from keras.models import Sequential
    from keras.layers.core import TimeDistributedDense, Dense
    from keras.layers.recurrent import SimpleRNN, LSTM
    model = Sequential()
    model.add(TimeDistributedDense(input_shape=(steps, 27), output_dim=hidden[0], activation='relu'))
    model.add(SimpleRNN(output_dim=hidden[1], activation='tanh'))
    model.add(Dense(output_dim=12, activation='tanh'))
    model.compile(optimizer='adam', loss='mse')
    return model

def plot(y, p):
    f, (ax0, ax1, ax2, ax3, ax4, ax5) = plt.subplots(6, 1, sharey=True)
    ax0.plot(y[:, 0])
    ax1.plot(y[:, 2])
    ax2.plot(y[:, 4])
    ax3.plot(y[:, 6])
    ax4.plot(y[:, 8])
    ax5.plot(y[:, 10])
    ax0.plot(p[:, 0])
    ax1.plot(p[:, 2])
    ax2.plot(p[:, 4])
    ax3.plot(p[:, 6])
    ax4.plot(p[:, 8])
    ax5.plot(p[:, 10])
    plt.show()


def time_distribute(data, steps):
    def shift(s_mat, s_vec):
        s = np.zeros_like(s_mat)
        s[:-1, :] = s_mat[1:, :]
        s[-1, :] = s_vec
        return s
    x = np.zeros((steps, len(data[0])))
    X = []
    for i in range(len(data[:,0])):
        x = shift(x, data[i])
        X.append(x)
    return np.array(X)


def ddpg_model(steps, hidden):
    alpha, beta, gamma, tau = 1e-3, 1e-3, 0.99, 1e-3
    consts = alpha, beta, gamma, tau

    model = DDPG(
        rnn_steps=steps,
        dt=1e-2,
        batch_size=64,
        hidden_a=hidden,
        hidden_c=[],
        consts=consts,
        s_dim=27,
        a_dim=12)

    # if is_rnn:
    #     print model.fwp_actor(np.zeros((1,steps,27), dtype=theano.config.floatX))
    # else:
    #     print model.fwp_actor(np.zeros((1,27), dtype=theano.config.floatX))

    return model

def shuffle_(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)



def load_and_train():
    x = np.load('data/playbackx_pos.npy')
    y = np.load('data/playbacky_pos.npy')

    # x = np.load('data/playbackx_vel.npy')
    # y = np.load('data/playbacky_vel.npy')

    x = np.transpose(x[:, :-1])[0:250]
    y = np.transpose(y[:, 1:])[0:250]
    s_orig = np.array(x, dtype=theano.config.floatX)
    a_orig = np.array(y, dtype=theano.config.floatX) / 2.

    print theano.config.floatX

    models = ([],[10],[10,10],[50],[50,50],
              [100,100],['lstm'],['lstm',10],['lstm',100],
              ['rnn'],['rnn',10],['rnn',100])

    for ix in range(10):
        for layers in models:
            s = np.copy(s_orig)
            a = np.copy(a_orig)
            steps = 1
            is_rnn = any([isinstance(x, str) for x in layers])
            if is_rnn:
                steps = 10
                s = time_distribute(s, steps)
            model = ddpg_model(steps, layers)
            for i in range(1000):
                shuffle_(s, a)
                loss = model.train_actor_supervised(s, a)
            name = '_'.join([str(x) for x in layers])+'_'+str(ix)

            print 'model=',name,', loss=', loss
            for par, par_ in zip(model.actor.nn.params, model.actor_.nn.params):
                par_.set_value(par.get_value())
            save_ddpg(model, 'models/supervised_ddpg_'+name+'.model')


def show_results():
    class C:
        def __init__(self, key, vals):
            self.key = key
            self.vals = vals
            self.mean = np.mean(np.array(vals))
            self.min = min(vals)
            self.max = max(vals)

    import pickle
    results = pickle.load(open('par_search_results.pkl'))
    results = [C(key, results[key]) for key in results.keys()]
    results.sort(key=lambda x: x.max, reverse=True)
    for c in results:
        print c.key, '===', c.mean, '(',c.min,')', c.max


def plot_data():
    x = np.load('data/playbackx_pos.npy')
    y = np.load('data/playbacky_pos.npy')
    x = np.transpose(x[:, :-1])[0:250]
    y = np.transpose(y[:, 1:])[0:250]
    s_orig = np.array(x, dtype=theano.config.floatX)
    a_orig = np.array(y, dtype=theano.config.floatX) / 2.

    y2 = np.load('data/rnn_10_learnedy.npy')[1:251]
    # y2 = np.transpose(y[:, 1:])[0:250]
    # plt.plot(y2)
    # plt.show()
    plot(a_orig, y2)

# plot_data()
show_results()