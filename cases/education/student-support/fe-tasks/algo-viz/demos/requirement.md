# Requirement

利用nm.js,nelder-mead算法的可视化代码重点完成一下内容：

1. 开发测试函数的等线图和整合功能
2. 缩放和显示坐标那些
3. 三个算法 然后起点可以自己定 然后坐标可以缩放和显示
1.可视化另外3种算法优化路径, SGD with Momentum, RMSProp,Adam 我会提供这些算法的代码, Nelder-mead 算法,这个算法我会提供完整的可视化代码
2. 还需要开发4种测试函数的等线图 Rastrigin function, Rosenbrock function, Booth function, Beale's function. 可以更改测试方程,但需要提前交流
3. 整合这些算法和测试方程, 例如可以选择不同的测试方程等线图,以及选择哪些算法进行优化路径可视化

当前实现的是： 源代码里实现的是SGD算法和himmelblau测试函数

这是SGD with Momentum的算法代码:
```js
function get_momentum_path(x0, y0, learning_rate, num_steps, momentum) { 
 var v_x = 0, 
 v_y = 0; 
 var momentum_history = [{"x": scale_x.invert(x0), "y": scale_y.invert(y0)}]; 
 var x1, y1, gradient; 
 for (i=0; i < num_steps; i++) { 
 gradient = grad_f(x0, y0) 
 v_x = momentum * v_x - learning_rate * gradient[0] 
 v_y = momentum * v_y - learning_rate * gradient[1] 
 x1 = x0 + v_x 
 y1 = y0 + v_y 
 momentum_history.push({"x" : scale_x.invert(x1), "y" : scale_y.invert(y1)}) 
 x0 = x1 
 y0 = y1 
 } 
 return momentum_history 
 } 
 ```

这是RMSProp的算法代码:
```js
 unction get_rmsprop_path(x0, y0, learning_rate, num_steps, decay_rate, eps) { 
 var cache_x = 0, 
 cache_y = 0; 
 var rmsprop_history = [{"x": scale_x.invert(x0), "y": scale_y.invert(y0)}]; 
 var x1, y1, gradient; 
 for (i = 0; i < num_steps; i++) { 
 gradient = grad_f(x0, y0) 
 cache_x = decay_rate * cache_x + (1 - decay_rate) * gradient[0] * gradient[0] 
 cache_y = decay_rate * cache_y + (1 - decay_rate) * gradient[1] * gradient[1] 
 x1 = x0 - learning_rate * gradient[0] / (Math.sqrt(cache_x) + eps) 
 y1 = y0 - learning_rate * gradient[1] / (Math.sqrt(cache_y) + eps) 
 rmsprop_history.push({"x" : scale_x.invert(x1), "y" : scale_y.invert(y1)}) 
 x0 = x1 
 y0 = y1 
 } 
 return rmsprop_history; 
 } 
 ```


 ```js
 function get_adam_path(x0, y0, learning_rate, num_steps, beta_1, beta_2, eps) { 
 var m_x = 0, 
 m_y = 0, 
 v_x = 0, 
 v_y = 0; 
 var adam_history = [{"x": scale_x.invert(x0), "y": scale_y.invert(y0)}]; 
 var x1, y1, gradient; 
 for (i = 0; i < num_steps; i++) { 
 gradient = grad_f(x0, y0) 
 m_x = beta_1 * m_x + (1 - beta_1) * gradient[0] 
 m_y = beta_1 * m_y + (1 - beta_1) * gradient[1] 
 v_x = beta_2 * v_x + (1 - beta_2) * gradient[0] * gradient[0] 
 v_y = beta_2 * v_y + (1 - beta_2) * gradient[1] * gradient[1] 
 x1 = x0 - learning_rate * m_x / (Math.sqrt(v_x) + eps) 
 y1 = y0 - learning_rate * m_y / (Math.sqrt(v_y) + eps) 
 adam_history.push({"x" : scale_x.invert(x1), "y" : scale_y.invert(y1)}) 
 x0 = x1 
 y0 = y1 
 } 
 return adam_history; 
 } 
 ```