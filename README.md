#Simple Solution for KDD 2019 | Policy Learning for Malaria Elimination

## Chinese Version

### Feedback phase 
无

# GA-method
### Check phase
使用 DQN , GA发现算法收敛到500分左右，此时检查解发现最好的解应该是[[0,1],[1,0],[0,1],[1,0],[1,1]],用[0,1]交替的方法能够在这个阶段达到520分左右，在以[[0,1],[1,0],[0,1],[1,0],[1,1]]，[[1,0],[0,1],[1,0],[0,1],[1,1]]为遗传算法的初始点，在小范围进行搜索后发现效果并没有提升.分析之后认为撒药和发蚊帐的操作都会产生类似抗药性的效果，在实际情况下需要将这两种操作交替使用。
 
### Verification phase
1. 观察feedback可以发现，reward都是正负出现，但是显然负数出现并不是说明拥有负收益，所以我对reward求了一个前缀和，即reward[i] = reward[i-1] + reward[i].
2. 基于这样的reward，我发现Check phase发现的规律依然适用，类似抗药性的效果依然存在。由于最终测评的environment依然不可知，使用遗传算法做最后的提交比较稳妥。
3. 在使用一些技巧，突破环境只有100次交互的限制之后，我把遗传算法的性能调试到稳定收敛于500-600分，但是这并不能作为最终答案。我需要使用一些方法让我的遗传算法能够在100次交互也能收敛到类似的位置。
4. 在设计遗传算法的初始值时，我只取了5个点[ 0 , 0.2 , 0.4 , 0.6 , 0.8 , 1]，而没有使用全局随机
```
def make_random_individuals(x,y):
    value = np.random.choice(act_space , (x,y) )
    
    for i in range(x):
        for j in range(y // 2):
            if value[i][ j * 2] + value[i][ j * 2 + 1] > 1.4 or value[i][ j * 2] + value[i][ j * 2 + 1] <= 0.2:
                if np.random.rand(1) > 0.5:
                    value[i][j * 2 + 1] = 1 - value[i][j * 2]
                else:
                    value[i][j * 2 ] = 1 - value[i][j * 2 + 1]
    return value

```
6. 在mutate操作中，我限制了每个点的搜索范围，再使用了Check phase中得到的insight，设计了一种1-x的操作，具体见code
```
def mutate(chromosome):
    mutation_rate = .5
    for j in range(chromosome.shape[0] // 2):
        left = j * 2
        right = j * 2 + 1
        r = np.random.rand(1);
        if(r > mutation_rate):
            mutetype = np.random.rand(1)
            if mutetype > 0.5:
                chromosome[left] = np.remainder(chromosome[left]+np.random.randn(1) * 0.4 ,0.99);
                chromosome[right] = np.remainder(chromosome[right]+np.random.randn(1) * 0.4 ,0.99);
            else:
                r2 = np.random.rand(1) 
                if ( r2 > 0.5):
                    chromosome[right] = 1 - chromosome[right]
                else:
                    chromosome[left] = 1 - chromosome[left]
    return chromosome
```

7. 同样为了接近check phase中得到的结果，我把cross操作修改成如下
```
def crossover(a,b):
     
      cross_point = int((self.num_params // 2)*np.random.rand(1)) * 2 - 1;
      c = np.append(a[:cross_point], b[cross_point:self.num_params]);
      return c
```

# Q-learning Method
To-do

# 1000 Epoch Q-learning Method(can't submit as answer)
To-do

# Tricky submission

## Policy
[[0,1],[1,0],[0,1],[1,0],[1,1]]
[[1,0],[0,1],[1,0],[0,1],[1,1]]

## Code 
see 0-1test.ipynb

## Specific Statement
To-do