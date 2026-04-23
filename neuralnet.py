import numpy as np
class Neural:
  def __init__(self,input_size=784,hidden_layers=[128,128],output_size=10):
    self.input_size = input_size
    self.hidden_layers = hidden_layers
    self.output_size = output_size
    self.weights = []
    self.biases = []
    self.layers = [] # Post-ReLU values
    self.layers_z = [] #Pre-ReLU values

    #Input for hidden layers
    self.weights.append(0.1 * np.random.randn(input_size,hidden_layers[0]))
    self.biases.append(np.zeros((1,hidden_layers[0])))

    #Hidden Layers Network
    for i in range(len(hidden_layers)-1):
      self.weights.append(0.1 * np.random.randn(hidden_layers[i],hidden_layers[i+1]))
      self.biases.append(np.zeros((1,hidden_layers[i+1])))

    #Hidden layers to Output
    self.weights.append(0.1 * np.random.randn(hidden_layers[-1],output_size))
    self.biases.append(np.zeros((1,output_size)))

  def forward(self,inputs):
    self.layers = [inputs]
    self.layers_z = []
    #Dot product of inputs and weights
    for i in range(len(self.weights)):
      if i == (len(self.weights)-1):
        self.layers_z.append(np.dot(self.layers[-1],self.weights[i])+self.biases[i])
      else: 
        output = np.dot(self.layers[-1],self.weights[i])+self.biases[i]
        self.layers_z.append(output)
        output = self.relu(output)
        self.layers.append(output)
    self.layers.append(self.softmax_act(self.layers_z[-1]))
    return self.layers[-1]

  def relu(self,inputs):
    return np.where(inputs < 0, 0, inputs)

  def softmax_act(self,inputs):
    stable_input = inputs - np.max(inputs,axis=1,keepdims=True)
    exp_values = np.exp(stable_input)
    return exp_values / np.sum(exp_values,axis=1,keepdims=True)

  def loss(self,input,target,epsilon=1e-15):
    # Clip predictions to avoid numerical instability (log(0))
    y_pred = np.clip(input, epsilon, 1 - epsilon)
    
    # Calculate loss for each sample and then the average over the batch
    loss = -np.mean(np.sum(target * np.log(y_pred), axis=-1))
    return loss
  
  def backpass(self,inputs,target,batch_size):
    dOutput = np.subtract(self.layers[-1],target)
    dgradient = dOutput/batch_size
    d_weights_list = []
    d_biases_list = []
    for i in range(len(self.weights)-1,-1,-1):
      dweights = np.dot(self.layers[i].T, dgradient)
      dbiases = np.sum(dgradient,axis=0)
      dgradient = np.dot(dgradient, self.weights[i].T)
      if i != (len(self.weights)-1) and i!=0:
        dgradient = np.multiply(dgradient,self.layers_z[i-1]>0)
      d_weights_list.append(dweights)
      d_biases_list.append(dbiases)
    d_weights_list = d_weights_list[::-1]
    d_biases_list = d_biases_list[::-1]
    return d_weights_list, d_biases_list

  def update(self, d_weights_list, d_biases_list, learning_rate=0.01):
    for i in range(len(self.weights)):
        self.weights[i] -= learning_rate * d_weights_list[i]
        self.biases[i]  -= learning_rate * d_biases_list[i]

  def train(self,inputs,target,batch_size,epochs,learning_rate):
    for i in range(epochs):
      print(f"Epoch {i}")
      indices = np.random.permutation(len(inputs))
      r_inputs = inputs[indices]
      r_target = target[indices]
      losses = []
      for j in range(0,len(inputs),batch_size):
        batch_inputs = r_inputs[j:j+batch_size]
        batch_target = r_target[j:j+batch_size]
        f_pass = self.forward(batch_inputs)
        loss = self.loss(f_pass,batch_target)
        losses.append(loss)
        d_w , d_l = self.backpass(batch_inputs,batch_target,batch_size)
        self.update(d_w,d_l,learning_rate)
      print(f"Loss : {np.mean(losses)}")

  def save_weights(self):
    np.savez('weights.npz', weights0 = self.weights[0], weights1 = self.weights[1],weights2 = self.weights[2])
    np.savez('biases.npz', biases0 = self.biases[0], biases1 = self.biases[1],biases2 = self.biases[2])
    
  def load_weights(self):
    weight = np.load('weights.npz')
    bias = np.load('biases.npz')
    self.weights = []
    self.biases = []
    for i in range(len(self.hidden_layers)+1):
      self.weights.append(weight[f'weights{i}'])
      self.biases.append(bias[f'biases{i}'])
  
def true_vals(array):
  array = array.flatten()
  true_values = np.zeros((array.size,array.max()+1))
  true_values[np.arange(array.size),array] = 1
  return true_values

def accuracy(y_pred,y_true):
    pred = np.argmax(y_pred,axis=1)
    true = np.argmax(y_true,axis=1)
    count = (pred == true).sum()
    return count/len(y_true)

