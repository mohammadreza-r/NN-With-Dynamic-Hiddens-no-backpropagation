import NN_With_Dynamic_Hiddens.NeuralNetwork as NeuralNetwork

# NN With one hidden layer
ann_test1=NeuralNetwork(3,[3],3,0.1)
print(ann_test1.query([1,2,5]))
#output : [[0.48998566],[0.73743379],[0.41439276]]

# NN With two hidden layers
ann_test2=NeuralNetwork(3,[3,3],3,0.1)
print(ann_test2.query([1,2,5]))
#output : [[0.35135697],[0.41949327],[0.4691811 ]]

# NN With three hidden layers
ann_test3=NeuralNetwork(3,[3,3,3],3,0.1)
print(ann_test3.query([1,2,5]))
#output : [[0.42100255],[0.31441832],[0.61506946]]

# NN With four hidden layers
ann_test4=NeuralNetwork(3,[3,3,3,3],3,0.1)
print(ann_test5.query([1,2,5]))
#output : [[0.18513424],[0.61977295],[0.59894078]]

# NN With five hidden layers
ann_test5=NeuralNetwork(3,[3,3,3,3,3],3,0.1)
print(ann_test5.query([1,2,5]))
#output : [[0.57725221],[0.4016175 ],[0.70397186]]
