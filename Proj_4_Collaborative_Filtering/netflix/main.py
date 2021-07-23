import numpy as np
import kmeans
import common
import naive_em
import em as em

X = np.loadtxt("toy_data.txt")

# TODO: Your code here
K=[1,2,3,4]
seed = [0,1,2,3,4]

def print_k_means_plots_and_cost(X,K,seed):
    for i in K:
        for j in seed:
            mixture,post = common.init(X, i, j)
            mixture, post, cost = kmeans.run(X,mixture,post)
            #plot
            #common.plot(X, mixture, post,'K: '+str(i)+', seed: '+str(j)+', Cost: '+str(cost))
            #print string result
            #print('K: '+str(i)+', seed: '+str(j)+', Cost: '+str(cost))
            return X,mixture,post


def print_EM_plots_and_cost(X,K,seed):

    for i in K:
        for j in seed:
            mixture,post = common.init(X, i, j)
            mixture, post, cost = naive_em.run(X,mixture,post)

            #plot
            #common.plot_mine(X, mixture, post,'K: '+str(i)+', seed: '+str(j)+', Cost: '+str(cost))
            #print string result
            print('K: '+str(i)+', seed: '+str(j)+', Cost: '+str(cost) +', seed: '+ str(seed_min))
            #return X,mixture,post


def print_kandEM_plots_and_cost(X,K,seed):
    for i in K:
        for j in seed:
            mixture,post = common.init(X, i, j)
            mixture, post, cost = kmeans.run(X,mixture,post)
            mixture2, post2, cost2 = naive_em.run(X, mixture, post)
            common.plot_mine(X, X, mixture, mixture2, post, post2, 'KM', 'EM', 'K: '+str(i)+', seed: '+str(j)+', Cost KM: '+str(cost)+', Cost EM: '+str(cost2))

#print_kandEM_plots_and_cost(X,K,seed)


def EM_BIC(X, K, seed):
    for i in K:
        for j in seed:
            mixture, post = common.init(X, i, j)
            mixture, post, cost = naive_em.run(X, mixture, post)
            bic = common.bic(X,mixture,cost)
            print('K: ' + str(i) + ', seed: ' + str(j) + ', Cost: ' + str(cost) + ', bic: ' + str(bic))


#EM_BIC(X, K, seed)

def print_EM_Mcompletion(X,K,seed):
    for i in K:
        for j in seed:
            mixture,post = common.init(X, i, j)
            mixture, post, cost = em.run(X,mixture,post)
            #print('K: ' + str(i) + ', seed: ' + str(j) + ', Cost: ' + str(cost))
            return mixture

X = np.loadtxt("netflix_incomplete.txt")
#K=[12]
#seed = [0,1,2,3,4]

#print_EM_Mcompletion(X,K,seed)

K=[12]
seed = [1]
mixture_12 = print_EM_Mcompletion(X,K,seed)
X_pred = em.fill_matrix(X, mixture_12)

X_gold = np.loadtxt('netflix_complete.txt')
print(common.rmse(X_gold,X_pred))