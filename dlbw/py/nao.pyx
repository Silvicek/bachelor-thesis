import numpy as np
from libcpp.vector cimport vector
from rnn_orig import load_model




step = 0
A = []
x = []
y = []

bestA = []
bestscore = -9999.
episodes = 100

model = []
keras_holder = []

cdef timesteps = 10
###########################################################################
cdef public int nao_init():
    global episodes, A
    A = np.random.normal(0,0.1,(27,12))
    print 'print from nao_init'
    return episodes

cdef public vector[float] nao_step(vector[float] x):
    global A
#    print 'print from nao_step'
    out = np.dot(np.array(x),A)
#    out = np.array([-1]*len(x))
    return out

cdef public void nao_end_episode(float score):
    global A,bestA,bestscore
    if score > bestscore:
        bestscore = score
        bestA = A
    A = np.random.normal(0,0.1,(27,12))  

cdef public void nao_finish():
    global bestscore, A
    print 'saving array, score =',bestscore
    np.save("programming/dlbw/models/A"+str(bestscore)+".model",bestA)
########################################################################### TEST
s = []
recording = [] #[x,s,o]
cdef public int nao_test_init():
    global model, s, recording
    model = load_model('programming/dlbw/py/data/rnn')
    s = np.zeros(model.hidden_dim)
    recording = [np.zeros((1000,model.in_dim)),np.zeros((1000,model.hidden_dim)),np.zeros((1000,model.out_dim))]
    print 'nao_test_init initialized'
    return 100
   
cdef public vector[float] nao_test_step(vector[float] x):
    global model, s, step, recording
    wait = 10
    if step > wait:
        out, s = model.step(np.array(x),s)
        if (step-wait < len(recording[0])):
            recording[0][step-wait][:] = np.array(x)
            recording[1][step-wait][:] = s
            recording[2][step-wait][:] = out
    else:
        out = np.zeros(12)
    step += 1
    return out

cdef public void nao_test_end_episode(float score):
    global s, model, step, recording
    s = np.zeros(model.hidden_dim)
    step = 0
    np.save("programming/dlbw/py/data/test_recording_x",recording[0])
    np.save("programming/dlbw/py/data/test_recording_s",recording[1])
    np.save("programming/dlbw/py/data/test_recording_o",recording[2])
###########################################################################

cdef public void nao_playback_init():
    global x,y
    x = np.zeros((27,1400))
    y = np.zeros((12,1400))
    print 'print from nao_playback_init'
    
cdef public void nao_playback_step(vector[float] a, vector[float] b):
    global x,y,step
    if step < 1400:
        x[:,step] = np.array(a)
        y[:,step] = np.array(b)
    step += 1

cdef public void nao_playback_finish():
    global step,x,y
    print 'saving array, steps =',step
    np.save("programming/dlbw/py/data/playbackx",x)
    np.save("programming/dlbw/py/data/playbacky",y)
    
###########################################################################
fixed_data = []
cdef public int nao_fixed_init():
    global fixed_data
    fixed_data = np.load('programming/dlbw/py/data/fixed.npy')
    print 'nao_fixed_init initialized'
    return 10
 
cdef public void nao_fixed_end_episode(float score):
    global step
    step = 0
    
cdef public vector[float] nao_fixed_step(vector[float] x):
    global fixed_data, step
    if step > len(fixed_data):
        return np.zeros(len(fixed_data[0]))
    out = fixed_data[step]
    step += 1
    return out

############################################################################ LEARN
dJdU = []
dJdV = []
dJdW = []
parameters = 0
episode = 0
updates = 0
J = 0
e = 0

cdef public int nao_learn_init():
    global model, s, parameters, updates, e, dJdU, dJdV, dJdW
    model = load_model('programming/dlbw/py/data/rnn')
    s = np.zeros(model.hidden_dim)
    parameters = model.V.size #+ model.V.size + model.W.size
    dJdU = np.zeros(model.U.shape)
    dJdV = np.zeros(model.V.shape)
    dJdW = np.zeros(model.W.shape)
    updates = 100
    e = 5e-3
    return (parameters+1) * updates + 2

cdef public vector[float] nao_learn_step(vector[float] x):
    global model, s, step
    if step > 10:
        out, s = model.step(np.array(x),s)
    else:
        s = np.zeros(model.hidden_dim)
        out = np.zeros(12)
    step += 1
    return out

def indexes(episode):
    a,b = model.V.shape
    j = episode % b
    i = (episode-j)/b
    nextj = (episode+1) % b
    nexti = (episode+1-j)/b
    return (i,j,nexti,nextj)

cdef public void nao_learn_end_episode(float score):
    global model, step, episode, parameters, e, J, dJdV, updates
    if J == 0:
        J = score
        e *= -1
        print 'BASE SCORE =',score
        model.save_model("programming/dlbw/py/data/rnn"+'_'+str(updates)+'_'+str('%.2f' % (score)))
        model.V[0][0] += e
        step = 0
        return
    if episode > parameters-1:
        model.V[-1][-1] -= e
        model.V += 1e-5 * dJdV
        dJdV = np.zeros(model.V.shape)
        J = 0
        episode = 0
        updates -= 1
    (i,j,nexti,nextj) = indexes(episode)
    model.V[i][j] -= e
    model.V[nexti][nextj] += e
    dJdV[i][j] = (score - J)/e
    step = 0
    episode += 1
    
###################################PARAMETER_SEARCH - ddpg
from ddpg.ddpg import DDPG, Holder, save_ddpg, load_ddpg, shift
import pickle
suff_list = []
index = -1
repeat = 9
import glob
par_search_results = {}
cdef public void nao_par_search_end_episode(float score):
    global s, model, step, index, suff_list, repeat, par_search_results

    if index == -1:
        w_list = glob.glob("programming/dlbw/py/models/*")
        suff_list = [x.split('/')[-1] for x in w_list]
        suff_list.sort()
    else:
        print suff_list[index], 'SCORE=', score
        par_search_results[suff_list[index]].append(score)
        pickle.dump(par_search_results, open('programming/dlbw/py/par_search_results.pkl','wb'))

    if repeat == 9:
        repeat = 0
        index += 1
        par_search_results[suff_list[index]] = []
        model = load_ddpg('programming/dlbw/py/models/'+suff_list[index])
    else:
        repeat += 1

    s = np.zeros((model.rnn_steps,27))
    step = 0



############################################################################ DDPG
sat = np.pi
s, a = 0., 0.
score = 0.
history = [[],[]]
bestscore = -9999.
last_distance = 0.
cdef public int nao_ddpg_init():
    global model, episode, s
    print 'starting ddpg'
    alpha, beta, gamma, tau = 1e-3, 1e-4, 0.99, 1e-3
    consts = alpha, beta, gamma, tau
    # model = DDPG(dt=5e-2,
    #              batch_size=64,
    #              hidden_a=['rnn',10],
    #              hidden_c=[200, 200],
    #              consts=consts,
    #              s_dim=27,
    #              a_dim=12,
    #              noise_theta=0.15, noise_sigma=0.1, noise_T=5.,
    #              rnn_steps=10
    #              )

    model = load_ddpg('programming/dlbw/models/ddpg_start.model')
    # from ddpg.explorers import OrnsteinUhlenbeck
    # model.noise = OrnsteinUhlenbeck(model.a_dim, dt=model.dt, T=5.,
    #                                 theta=.15, sigma=.05)

    s = np.zeros((model.rnn_steps,27))

    episode = 0
    import theano
    print 'DDPG MODEL INITIALIZED', theano.config.floatX
    updates = 1000000
    return updates

cdef public vector[float] nao_ddpg_step(vector[float] x, float p):
    global model, holder, step, episode, s, a, score, last_distance

    s_t = np.array(x).astype('float64').clip(-10.,10.)
    r = model.dt*(.1*(1.-np.mean(np.abs(a)))+100*(p-last_distance))
    # r = model.dt*(1-.5*np.mean(np.abs(a))-.5*np.mean(np.abs(s_t)))


    if model.is_rnn:
        s = shift(s, s_t)
    else:
        s = s_t

    s_ = np.expand_dims(s, 0)

    if episode % 50 == 0:
        if step > 2:
            a = model.fwp_actor_(s_)
        else:
            a = np.zeros((1, 12)).astype('float64')
        print 'mean_s=', np.mean(np.abs(s_[-1]))
        print 'mean_a=',np.mean(np.abs(a))
        print 'reward=',r
        print 'position=',p
        print '==========> critic=', model.fwp_critic(np.expand_dims(s_t, 0),a)[0,0]

    else:
        a = np.zeros(12).astype('float64')
        if step > 2:
            a = model.stoch_action(s_, 1.)
            holder.complete(np.float64(r), s)
            model.R_add(holder)
            holder = Holder(s, a.flatten())
            model.train(prioritize=True)
        elif step == 2:
            holder = Holder(s, a.flatten())
            last_distance = 0.
    step += 1
    # print 'r=',r-R, 'A=', a.flatten() * sat
    score += r
    last_distance = p
    return a.flatten()

cdef public void nao_ddpg_end_episode(float t):
    global step, episode, model, s, a, score, history, bestscore
    if t > 14:
        r = 0*model.dt*(1-.1*np.mean(np.abs(a)))
    else:
        r = -.1
    holder = Holder(s, a.flatten(), last=True)
    holder.complete(np.float64(r), s)
    model.R_add(holder)
    model.noise.reset()
    s = np.zeros((model.rnn_steps,27))

    step = 0
    print '%d. EPISODE SCORE = %.3f' % (episode, score)
    if episode % 50 == 0:
        history[0].append(model.time_steps)
        history[1].append(score)
        np.save(open('programming/dlbw/models/plot.npy', 'wb'), np.array(history))

        if score > bestscore:
            bestscore = score
            save_ddpg(model, 'programming/dlbw/models/ddpg_best.model')
            print 'saving', 'ddpg_best.model'

    if episode % 1000 == 0:
        save_ddpg(model, 'programming/dlbw/models/ddpg_'+str(episode)+'.model')
        print 'saving', 'ddpg_'+str(episode)+'.model'

    episode += 1
    score = 0.


####################################################################
data = []
cdef public int nao_ddpg_test_init():
    global model, s
    print 'printing from nao_ddpg_test_init'
    model = load_ddpg('programming/dlbw/models/ddpg_best.model')
    # model = load_ddpg('programming/dlbw/py/supervised_ddpg.model')
    s = np.zeros((model.rnn_steps,27))
    print 'DDPG MODEL LOADED'
    updates = 10
    return updates

cdef public vector[float] nao_ddpg_test_step(vector[float] x):
    global model, holder, step, s, data
    s_t = np.array(x).astype('float64').clip(-10.,10.)
    if model.is_rnn:
        s = shift(s, s_t)
    else:
        s = s_t
    if step > 2:
        a = model.fwp_actor_(np.expand_dims(s,0))
        # data.append((s_t,a.flatten()))
    else:
        a = np.zeros(12).astype('float64')
    step += 1
    a = a.flatten()
    print 's=', s
    print 'a=', a
    return a

cdef public void nao_ddpg_test_end_episode(float score):
    global step, episode, s, data
    step = 0
    print 'nao_ddpg_test_end_episode'
    print '%d. EPISODE SCORE = %.3f' % (episode, score)
    # x = np.array([s for s,a in data])
    # y = np.array([a for s,a in data])
    # np.save("programming/dlbw/py/data/rnn_10_learnedx",x)
    # np.save("programming/dlbw/py/data/rnn_10_learnedy",y)

    state = np.zeros_like(s)
    episode += 1