
import numpy as np
import matplotlib . pyplot as plt


# et chaque dimension : faux , vrai 7

prob_pluie = np.array([0.8, 0.2]).reshape(2, 1, 1, 1)
print ("Pr(Pluie)=\n{}".format(np.squeeze(prob_pluie)))
print("\n")
prob_arroseur = np.array([0.9, 0.1]).reshape(1, 2, 1, 1)
print ("Pr(Arroseur)=\n{}".format(np.squeeze(prob_arroseur))) 
print("\n")


watson = np.array([[0.8, 0.2], [0, 1]]).reshape(2, 1, 2, 1)
print ("Pr(Watson|Pluie)=\n{}".format(np.squeeze(watson))) 
print("\n")

holmes = np.array([[[1, 0], [0.1, 0.9]], [[0, 1], [0, 1]]]).reshape(2, 2, 1, 2)
print ("Pr(Holmes|Pluie ,arroseur)=\n{}".format(np.squeeze(holmes)))
print("\n")

watson[0,:,1,:]


(watson * prob_pluie).sum(0).squeeze()[1]

holmes[0,1,0,1] 

#Question 1
#a) Pr(W = 1)
prob_w_p = watson * prob_pluie
print("p={}\n".format(prob_w_p))
prob_w = (prob_w_p).sum(0)
print("Q1 a) Pr(W = 1)={}\n".format(prob_w.squeeze()[1]))

#b) Pr(W = 1 | H = 1)
joint_prob_h_p_a = (holmes * prob_pluie * prob_arroseur)
prob_h = joint_prob_h_p_a.sum(1).sum(0).squeeze()[1]
print("Pr(H = 1)={}\n".format(prob_h))

joint_prob_w_h_p_a = watson * holmes * prob_pluie * prob_arroseur
prob_w_h = joint_prob_w_h_p_a.sum(1).sum(0).squeeze()[1][1] / prob_h
print("Q1 b) Pr(W = 1 | H = 1)={}\n".format(prob_w_h))

#c) Pr(W = 1 | H = 1, A = 0)
joint_prob_w_h_p = (watson * holmes * prob_pluie)
prob_w_h_a = joint_prob_w_h_p.sum(0).squeeze()[0][1][1]
joint_prob_h_p = (holmes * prob_pluie)
prob_w_h_a = prob_w_h_a / joint_prob_h_p.sum(0).squeeze()[0][1]
print("Q1 c) Pr(W = 1 | H = 1, A=0)={}\n".format(prob_w_h_a))

#d) Pr(W = 1 | A = 0)
joint_prob_w_p = (watson * prob_pluie)
prob_w_a = joint_prob_w_p.sum(0).squeeze()[1]
print("Q1 d) Pr(W = 1 | A = 0)={}\n".format(prob_w_a))

#e) Pr(W = 1 | P = 1)
print("Q1 e) Pr(W = 1 | P = 1)={}\n".format(watson.squeeze()[1][1]))





