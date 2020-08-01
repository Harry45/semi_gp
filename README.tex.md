# fractals
Mandelbulb fractal 3D in pure Javascript based on [GPU.JS](https://github.com/gpujs/gpu.js)

[Run Mandelbulb!](https://kamil-kielczewski.github.io/fractals/mandelbulb.html)

## Calc Ray

Input parameters ([left-handed coordinate system](https://en.wikipedia.org/wiki/Cartesian_coordinate_system#In_three_dimensions)): 
* $E = [Ex,Ey,Ez]$ - camera (eye) position at point 
* $T= [Tx,Ty,Tz]$ - target point where camera looks  
* $w=[wx,wy,wz]$ - camera vertical normalized vector which idicates where is up and were is down (not shown on picture, usually equal [0,1,0]). 
* $\theta \in [0,\pi)$ - field of view (slacar value, for human eye $\approx 90^\circ$)
* $k$ - number of pixels on screen width 
* $m$ - number of pixels screen in height 
