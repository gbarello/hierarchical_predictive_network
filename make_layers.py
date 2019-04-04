import numpy as np
import tensorflow as tf

def square_err(x,y):
    return tf.reduce_sum(tf.pow(x - y,2))
def np_square_err(x,y):
    return np.mean(np.power(x - y,2))


class hpred_layer:
    def __init__(self,size,batch_size,compnonlin = tf.nn.relu,ffnonlin = tf.nn.relu,fbnonlin = tf.nn.relu,outnl = tf.nn.relu,prednl = lambda x:x,stop_fwd_grad = True,stop_bwd_grad = True):
        self.stop_bwd_grad = stop_bwd_grad
        self.stop_fwd_grad = stop_fwd_grad
        self.size = size
        self.batch_size = batch_size
        
        self.fb_to_comp = tf.Variable(tf.random_normal([size,size])/tf.sqrt(np.float32(size)))
        self.ff_to_comp = tf.Variable(tf.random_normal([size,size])/tf.sqrt(np.float32(size)))
        self.comp_to_comp = tf.Variable(tf.random_normal([size,size])/(10*tf.sqrt(np.float32(size))))
        self.comp_to_output = tf.Variable(tf.random_normal([size,size])/tf.sqrt(np.float32(size)))

        self.out_to_pred = []

        self.fb_in = []
        self.ff_in = []

        self.fbnl = fbnonlin
        self.ffnl = ffnonlin
        self.compnl = compnonlin
        self.outnl = outnl
        self.prednl = prednl
        
        #self.fb_tensors = []
        #self.fw_tensors = []
        
    def add_ffwd_input(self,ffwd_size):
        self.ff_in.append(tf.Variable(tf.random_normal([self.size,int(ffwd_size)]))/tf.sqrt(np.float32(int(ffwd_size))))
        self.out_to_pred.append(tf.Variable(tf.zeros([int(ffwd_size),self.size])))
        #self.fw_tensors.append(ffwd)
        
    def add_fdbk_input(self,fdbk_size):
        self.fb_in.append(tf.Variable(tf.random_normal([self.size,int(fdbk_size)]))/tf.sqrt(np.float32(int(fdbk_size))))
        #self.fb_tensors.append(fdbk)

    def apply_layer(self,activity,ff,fb):

        #feedback
        fba = tf.zeros([self.batch_size,self.size])
        for k in range(len(self.fb_in)):
            if self.stop_bwd_grad:#stop the gradients from the feedback
                temp = tf.stop_gradient(fb[k])
            else:
                temp = fb[k]
            
            fba += tf.tensordot(temp,self.fb_in[k],axes = [[-1],[-1]])
            
        fba = self.fbnl(fba)

        #feedforward input
        ffa = tf.zeros([self.batch_size,self.size])
        
        for k in range(len(self.ff_in)):
            if self.stop_fwd_grad:#stop the gradients from ffwd input
                temp = tf.stop_gradient(ff[k])
            else:
                temp = ff[k]
                
            ffa += tf.tensordot(temp,self.ff_in[k],axes = [[-1],[-1]])
        ffa = self.ffnl(ffa)
        
        aout = self.compnl(
            tf.tensordot(activity,self.comp_to_comp,axes = [[-1],[-1]])
            + tf.tensordot(fba,self.fb_to_comp,axes = [[-1],[-1]])
            + tf.tensordot(ffa,self.ff_to_comp,axes = [[-1],[-1]])
        )

        out = self.outnl(tf.tensordot(aout,self.comp_to_output,axes = [[-1],[-1]]))

        pred_out = [self.prednl(tf.tensordot(out,self.out_to_pred[k],axes = [[-1],[-1]])) for k in range(len(self.out_to_pred))]

        return {"activity":aout,"output":out,"feedback_activity":fba,"feedforward_activity":ffa,"prediction":pred_out}
    
    
class conv_hpred_layer:
    def __init__(self,size,batch_size,compnonlin = tf.nn.relu,ffnonlin = tf.nn.relu,fbnonlin = tf.nn.relu,outnl = tf.nn.relu,prednl = lambda x:x,stop_fwd_grad = True,stop_bwd_grad = True):
    
        self.stop_bwd_grad = stop_bwd_grad
        self.stop_fwd_grad = stop_fwd_grad
        self.size = size
        self.batch_size = batch_size
        
        self.fb_to_comp = tf.Variable(tf.random_normal([size,size])/tf.sqrt(np.float32(size)))
        self.ff_to_comp = tf.Variable(tf.random_normal([size,size])/tf.sqrt(np.float32(size)))
        self.comp_to_comp = tf.Variable(tf.random_normal([size,size])/(10*tf.sqrt(np.float32(size))))
        self.comp_to_output = tf.Variable(tf.random_normal([size,size])/tf.sqrt(np.float32(size)))

        self.out_to_pred = []

        self.fb_in = []
        self.ff_in = []

        self.fbnl = fbnonlin
        self.ffnl = ffnonlin
        self.compnl = compnonlin
        self.outnl = outnl
        self.prednl = prednl
        
        #self.fb_tensors = []
        #self.fw_tensors = []
        
    def add_ffwd_input(self,ffwd_size):
        self.ff_in.append(tf.Variable(tf.random_normal([self.size,int(ffwd_size)]))/tf.sqrt(np.float32(int(ffwd_size))))
        self.out_to_pred.append(tf.Variable(tf.zeros([int(ffwd_size),self.size])))
        #self.fw_tensors.append(ffwd)
        
    def add_fdbk_input(self,fdbk_size):
        self.fb_in.append(tf.Variable(tf.random_normal([self.size,int(fdbk_size)]))/tf.sqrt(np.float32(int(fdbk_size))))
        #self.fb_tensors.append(fdbk)

    def apply_layer(self,activity,ff,fb):

        #feedback
        fba = tf.zeros([self.batch_size,self.size])
        for k in range(len(self.fb_in)):
            if self.stop_bwd_grad:#stop the gradients from the feedback
                temp = tf.stop_gradient(fb[k])
            else:
                temp = fb[k]
            
            fba += tf.tensordot(temp,self.fb_in[k],axes = [[-1],[-1]])
            
        fba = self.fbnl(fba)

        #feedforward input
        ffa = tf.zeros([self.batch_size,self.size])
        
        for k in range(len(self.ff_in)):
            if self.stop_fwd_grad:#stop the gradients from ffwd input
                temp = tf.stop_gradient(ff[k])
            else:
                temp = ff[k]
                
            ffa += tf.tensordot(temp,self.ff_in[k],axes = [[-1],[-1]])
        ffa = self.ffnl(ffa)
        
        aout = self.compnl(
            tf.tensordot(activity,self.comp_to_comp,axes = [[-1],[-1]])
            + tf.tensordot(fba,self.fb_to_comp,axes = [[-1],[-1]])
            + tf.tensordot(ffa,self.ff_to_comp,axes = [[-1],[-1]])
        )

        out = self.outnl(tf.tensordot(aout,self.comp_to_output,axes = [[-1],[-1]]))

        pred_out = [self.prednl(tf.tensordot(out,self.out_to_pred[k],axes = [[-1],[-1]])) for k in range(len(self.out_to_pred))]

        return {"activity":aout,"output":out,"feedback_activity":fba,"feedforward_activity":ffa,"prediction":pred_out}

    
class hpred:

    def __init__(self,data_shape,sizes,stop_fwd_grad = True,stop_bwd_grad = True):

        '''
        data_in = [batch_size,time,in_size]

        '''

        self.data_shape = data_shape
        self.data = tf.placeholder(tf.float32,self.data_shape,name = "data")
        
        self.layers = [hpred_layer(s,self.data_shape[0],stop_fwd_grad = stop_fwd_grad,stop_bwd_grad = stop_bwd_grad) for s in sizes]

        self.layers[0].add_ffwd_input(self.data_shape[-1])

        if len(sizes) > 1:
            self.layers[0].add_fdbk_input(sizes[1])
            self.layers[-1].add_ffwd_input(sizes[-2])
        
        for k in range(1,len(self.layers) - 1):
            self.layers[k].add_ffwd_input(sizes[k-1])
            self.layers[k].add_fdbk_input(sizes[k+1])
        
        self.init_activity = [tf.placeholder(tf.float32,[self.data_shape[0],s]) for s in sizes]
        self.init_outputs = [tf.placeholder(tf.float32,[self.data_shape[0],s]) for s in sizes]

        self.activities = [[self.init_activity[k]] for  k in range(len(self.init_activity))]
        self.outputs = [[self.init_outputs[k]] for  k in range(len(self.init_outputs))]

        self.layer_outputs = [[] for k in range(len(self.init_activity))]

        for t in range(int(self.data_shape[1])):
            for k in range(len(self.init_activity)):
                if k == 0:
                    self.layer_outputs[k].append(self.layers[k].apply_layer(self.activities[k][t],[self.data[:,t]],[self.activities[k+1][t]] if len(sizes) > 1 else []))
                elif k == len(self.init_activity) - 1 and len(sizes) > 1:
                    self.layer_outputs[k].append(self.layers[k].apply_layer(self.activities[k][t],[self.activities[k-1][t]],[]))
                else:
                    self.layer_outputs[k].append(self.layers[k].apply_layer(self.activities[k][t],[self.activities[k-1][t]],[self.activities[k+1][t]]))

                self.activities[k].append(self.layer_outputs[k][t]["activity"])
                self.outputs[k].append(self.layer_outputs[k][t]["output"])
                
        self.activities = [tf.stack(a,axis = 1) for a in self.activities]
        self.prediction = [tf.stack([t["prediction"][0] for t in a],axis = 1) for a in self.layer_outputs]
        self.final_activation = [a[:,-1] for a in self.activities]

    def get_attribute(self,att):
        out = [tf.stack([t[att] for t in k],axis = 1) for k in self.layer_outputs]
        return out
    
    def prediction_loss(self,f = square_err):

        out = self.get_attribute("activity")
        pred = self.prediction

        loss = [f(pred[0][:,:-1], self.data[:,1:])]
    
        for k in range(1,len(self.prediction)):
            print(self.prediction[k][0].shape)
            loss.append(f(pred[k][:,:-1],out[k-1][:,1:]))
            
        return tf.reduce_sum(loss),loss
        
        
if __name__ == "__main__":
    tot = 100000
    p = .01
    state = [0]
    tlen = 100

    trans = np.random.uniform(0,1,[tot])

    for k in trans:
        if k < p:
            state.append(1 - state[-1])
        else:
            state.append(state[-1])

    data = [0]

    rand = np.random.randn(tot)
    dec = .99
    
    for k in state:
        data.append(dec * data[-1] + (1. - dec)*(k * 2 - 1))

    data = np.expand_dims(np.expand_dims(np.array(data),0),-1)

    data += np.random.standard_normal(data.shape)*.01
    data = np.exp(data)


    test_frac = int(.9*tot)
    dat = data[:,:test_frac]
    var = data[:,test_frac:]

    isize = [1,tlen,1]
    
    A = hpred(isize,[10,10])

    for k in A.prediction:
        print(k.shape)

    loss,loss_list = A.prediction_loss()
    
    adam = tf.train.AdamOptimizer()

    train = adam.minimize(loss)
    
    final_activation = A.final_activation

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    final_act = [np.zeros([int(x) for x in k.shape],dtype = np.float32) for k in A.init_activity]

    for k in range(int(test_frac/tlen) - 1):
        repl_dict = {}
        for j in range(len(A.layers)):
            repl_dict[A.init_activity[j]] = final_act[j]


        repl_dict[A.data] = dat[:,tlen*k : tlen * (k+1)]
        LOSS = sess.run(loss_list,repl_dict)

        _,final_act = sess.run([train,A.final_activation],repl_dict)
        print("{}\t{}\t{}".format(round(float(k*tlen)/test_frac,2),LOSS,np.mean(state[tlen*k:tlen*(k+1)])))


    test_act = []
    for k in range(0,int((tot - test_frac)/tlen)):
        repl_dict = {}
        for j in range(len(A.layers)):
            repl_dict[A.init_activity[j]] = final_act[j]

        repl_dict[A.data] = var[:,tlen*k : tlen * (k+1)]

        final_act,stuff = sess.run([A.final_activation,A.activities],repl_dict)
        
        test_act.append(np.array(stuff)[:,0,1:])

    test_act = np.squeeze(np.concatenate(test_act,axis = 1))

    scor = np.array([np.corrcoef(test_act[-1,:,k],state[int(test_frac):-1]) for k in range(test_act.shape[2])])
    print(scor)
    
    scor = np.corrcoef(test_act[-1,:].transpose())
    print(scor)
    sess.close()
