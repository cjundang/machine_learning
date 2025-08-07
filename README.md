
# A Comprehensive Overview of Machine Learning Techniques: From Supervised Learning to Deep Neural Networks

## 2.1 Supervised Learning

Supervised learning involves training a model on a labeled dataset, where the target output (label) is provided. It consists of two main branches:

### A) Classification

#### 1. Logistic Regression

**Equation:**

$$
h_\theta(x) = \frac{1}{1 + e^{-\theta^T x}}
$$

**Python:**
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

#### 2. K-Nearest Neighbors (KNN)

**Equation:**

\[
d(x, x_i) = \sqrt{\sum_{j=1}^{n} (x_j - x_{ij})^2}
\]

**Python:**
```python
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
```

#### 3. Support Vector Machine (SVM)

**Equation:**

\[
\min \frac{1}{2} ||w||^2 \text{ subject to } y_i(w^T x_i + b) \geq 1
\]

**Python:**
```python
from sklearn.svm import SVC
model = SVC(kernel='linear')
model.fit(X_train, y_train)
```

#### 4. Decision Tree

**Metric (Entropy):**

\[
H(D) = - \sum_{i=1}^{k} p_i \log_2 p_i
\]

**Python:**
```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
```

#### 5. Naive Bayes

**Equation:**

\[
P(y|x) \propto P(y) \prod_{i=1}^n P(x_i|y)
\]

**Python:**
```python
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)
```

### B) Regression

#### 1. Linear Regression

**Equation:**

\[
y = \beta_0 + \beta_1 x_1 + \dots + \beta_n x_n + \epsilon
\]

**Python:**
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
```

#### 2. Polynomial Regression

**Python:**
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
model.fit(X_train, y_train)
```

#### 3. Ridge and Lasso Regression

**Ridge Equation:**

\[
\text{Loss} = MSE + \lambda \sum_{j=1}^n \beta_j^2
\]

**Lasso Equation:**

\[
\text{Loss} = MSE + \lambda \sum_{j=1}^n |\beta_j|
\]

**Python:**
```python
from sklearn.linear_model import Ridge, Lasso
ridge = Ridge(alpha=1.0).fit(X_train, y_train)
lasso = Lasso(alpha=0.1).fit(X_train, y_train)
```

## 2.2 Unsupervised Learning

### A) Clustering

#### 1. K-Means

**Objective:**

\[
\text{SSE} = \sum_{i=1}^{n} \sum_{k=1}^{K} \mathbb{1}(c_i = k) \|x_i - \mu_k\|^2
\]

**Python:**
```python
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)
model.fit(X)
```

#### 2. DBSCAN

**Python:**
```python
from sklearn.cluster import DBSCAN
model = DBSCAN(eps=0.5, min_samples=5)
model.fit(X)
```

#### 3. Agglomerative Clustering

**Python:**
```python
from sklearn.cluster import AgglomerativeClustering
model = AgglomerativeClustering(n_clusters=3)
model.fit(X)
```

#### 4. Mean Shift

**Python:**
```python
from sklearn.cluster import MeanShift
model = MeanShift()
model.fit(X)
```

#### 5. Fuzzy C-Means

**Python:**
```python
from fcmeans import FCM
fcm = FCM(n_clusters=3)
fcm.fit(X)
```

### B) Pattern Search

#### 1. Apriori

**Python:**
```python
from mlxtend.frequent_patterns import apriori
frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)
```

#### 2. FP-Growth

**Python:**
```python
from mlxtend.frequent_patterns import fpgrowth
frequent_itemsets = fpgrowth(df, min_support=0.2, use_colnames=True)
```

### C) Dimensionality Reduction

#### 1. PCA

**Python:**
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
```

#### 2. t-SNE

**Python:**
```python
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)
```

## 3. Ensemble Methods

### A) Bagging - Random Forest

**Python:**
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
```

### B) Boosting

#### 1. AdaBoost

**Python:**
```python
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier()
model.fit(X_train, y_train)
```

#### 2. Gradient Boosting

**Python:**
```python
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier()
model.fit(X_train, y_train)
```

#### 3. XGBoost

**Python:**
```python
import xgboost as xgb
model = xgb.XGBClassifier()
model.fit(X_train, y_train)
```

### C) Stacking

**Python:**
```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

estimators = [('svm', SVC()), ('lr', LogisticRegression())]
stack_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stack_model.fit(X_train, y_train)
```
 
## 4. Reinforcement Learning

Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by interacting with an environment.

### Key Elements:
- **Agent**: The learner or decision-maker.
- **Environment**: The external system the agent interacts with.
- **State (s)**: The current situation.
- **Action (a)**: The decision the agent makes.
- **Reward (r)**: The feedback received after an action.

---

### A) Q-Learning

**Update Rule:**

\[
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_t + \gamma \max_a Q(s_{t+1}, a) - Q(s_t, a_t) \right]
\]

**Python:**
```python
Q = np.zeros((n_states, n_actions))
for episode in range(episodes):
    state = env.reset()
    done = False
    while not done:
        action = np.argmax(Q[state] + np.random.randn(1, n_actions) * 0.01)
        next_state, reward, done, _ = env.step(action)
        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
        state = next_state
```

---

### B) SARSA

**Update Rule:**

\[
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [r_t + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)]
\]

---

### C) Deep Q-Network (DQN)

**Concept:** Replaces Q-table with a neural network to approximate Q-values.

**Python (Pseudo):**
```python
model = create_model()
for episode in range(episodes):
    state = env.reset()
    while not done:
        action = epsilon_greedy(state)
        next_state, reward, done = env.step(action)
        memory.append((state, action, reward, next_state, done))
        train_DQN(model, memory)
```

---

### D) REINFORCE (Policy Gradient)

**Gradient:**

\[
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t|s_t) R_t \right]
\]

**Python (Pseudo):**
```python
log_probs, rewards = [], []
for t in range(T):
    action = sample_action(state)
    log_probs.append(log_prob(action))
    rewards.append(reward)

loss = -sum([log_prob * R for log_prob, R in zip(log_probs, discounted_rewards)])
```

---

### E) Actor-Critic (A2C, A3C)

**Update:**

\[
\nabla_\theta J(\theta) \approx \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot (R_t - V(s_t))
\]

---

## 5. Neural Networks and Deep Learning

Deep Learning uses multiple layers of neurons to extract hierarchical features.

---

### A) Multi-Layer Perceptron (MLP)

**Equation:**

\[
a = f(Wx + b)
\]

**Python:**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

---

### B) Convolutional Neural Network (CNN)

**Equation:**

\[
S(i,j) = (X * K)(i,j) = \sum_m \sum_n X(i+m, j+n) K(m,n)
\]

**Python:**
```python
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten

model = Sequential([
    Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
```

---

### C) Recurrent Neural Network (RNN) and LSTM

**Equation (LSTM):**

\[
f_t = \sigma(W_f[h_{t-1}, x_t] + b_f) \\
i_t = \sigma(W_i[h_{t-1}, x_t] + b_i) \\
\tilde{C}_t = \tanh(W_C[h_{t-1}, x_t] + b_C) \\
C_t = f_t * C_{t-1} + i_t * \tilde{C}_t \\
o_t = \sigma(W_o[h_{t-1}, x_t] + b_o) \\
h_t = o_t * \tanh(C_t)
\]

**Python:**
```python
from tensorflow.keras.layers import LSTM, Embedding

model = Sequential([
    Embedding(input_dim=10000, output_dim=128),
    LSTM(64),
    Dense(1, activation='sigmoid')
])
```

---

### D) Autoencoder

**Loss Function:**

\[
L = \|x - \hat{x}\|^2
\]

**Python:**
```python
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

input_layer = Input(shape=(X.shape[1],))
encoded = Dense(64, activation='relu')(input_layer)
decoded = Dense(X.shape[1], activation='sigmoid')(encoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X, X, epochs=20)
```

---

### E) Generative Adversarial Network (GAN)

**Objective Function:**

\[
\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]
\]

**Python:**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU

generator = Sequential([
    Dense(128, input_dim=100),
    LeakyReLU(0.2),
    Dense(784, activation='tanh')
])

discriminator = Sequential([
    Dense(128, input_dim=784),
    LeakyReLU(0.2),
    Dense(1, activation='sigmoid')
])
```

---

## Summary

These methods offer a broad and flexible toolkit to address complex machine learning problems including cyber security, pattern recognition, anomaly detection, and predictive modeling.

