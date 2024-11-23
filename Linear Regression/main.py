import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("trainingset.csv")



#model is f_x = mx + b
#cost function is J_mb = 1/2n(mx + b - y)**2
#we have to find values for m and b sop that value of cost functions is a minimum: use gradient descent

def gradient_descent(m_temp, b_temp, data, L):
   
   m_gradient = 0
   b_gradient = 0


   n = len(data)
   for i in range(n):
      x = data.iloc[i]["size"]
      y = data.iloc[i]["price"]

      m_gradient = m_gradient + (1/n)*(m_temp * x + b_temp - y)*x
      b_gradient = b_gradient + (1/n)*(m_temp * x + b_temp - y)
   
   m = m_temp - m_gradient * L
   b = b_temp - b_gradient * L

   return m, b


# execution

m = 0
b = 0
L = 0.0000001
itr = 300

for i in range(itr):
   if i%50 == 0:
      print(f"Iteration is {i}")
   m , b = gradient_descent(m,b,data,L)

#m and b is found
print(m, b)

x_values = []
y_values = []
for i in range(800,1600):
   x_values.append(i)

for i in range(len(x_values)):
   y_values.append(m*x_values[i]+ b)


plt.plot(x_values,y_values,color = "red")
plt.scatter(data['size'],data['price'],color = "black")


plt.xlabel('House Size (Sq feet)')
plt.ylabel('Price $ (1000s)')
plt.show()
