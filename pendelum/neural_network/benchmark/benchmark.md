c) benchmark: stepsize = (pi / 16)
                score
    Network1:   0.7812
    Network2:   0.7500
    Network3:   0.9688
    Network4:   0.0625
    Network5:   0.6250
    Network6:   1
    Network7:   0
    Network8:   0.2500
    Network19:  0.3438
a) benchmark: stepsize = (pi / 32)
                score
    Network1:   1
    Network2:   0.9219
    Network3:   1
    Network4:   0.1094
    Network5:   0.6719
    Network6:   1
    Network7:   0
    Network8:   0.5156
    Network19:  0.6719
b) benchmark: stepsize = (pi / 64)
    Network1:  1
    Network2:  0.9688
    Network3:  1
    Network4:  0.2344
    Network5:  0.7109
    Network6:  1
    Network7:  0.1641
    Network8:  0.9297
    Network19: 0.7891



d) benchmark: singh stepsize = (pi / 16)
    Network1:  0.0312
    Network2:  0
    Network3:  0.3125
    Network4:  0
    Network5:  0.1875
    Network6:  0.0625
    Network7:  0
    Network8:  0
e) benchmark: poly_method = singh stepsize = (pi / 32)
    Network1:  0.1875
    Network2:  0.2188
    Network3:  0.9375
    Network4:  0.0625
    Network5:  0.3438
    Network6:  0.3125
    Network7:  0
    Network8:  0.0156
f) benchmark: poly_method = singh stepsize = (pi / 64)
    Network1:  0.6016
    Network2:  0.7656
    Network3:  1
    Network4:  0.2188
    Network5:  0.6250
    Network6:  1
    Network7:  0
    Network8:  0.2500
g) Bound approx --> False: singh stepsize = (pi / 64) TOOK MUCH LONGER!!
    Network1:  0.6016
    Network2:  0.7656
    Network3:  1
    Network4:  0.2188
    Network5:  0.6250
    Network6:  1
    Network7:  0
    Network8:  0.2500
h) Bound approx --> False: singh stepsize = (pi / 32) 
    Network1: 0.1875
    Network2: 0.2188
    Network3: 0.9375
    Network4: 0.0625
    Network5: 0.3438
    Network6: 0.3125
    Network7: 0
    Network8: 0.0156
i) Bound approx --> False: singh stepsize = (pi / 16) 
    Network1: 0.0312
    Network2: 0
    Network3: 0.3125
    Network4: 0
    Network5: 0.1875
    Network6: 0.0625
    Network7: 0
    Network8: 0
j)Bound approx --> True, num_generators=1000: singh stepsize = (pi / 16) , singh
    Network1: 0.0312
    Network2: 0
    Network3: 0.3125
    Network4: 0
    Network5: 0.1875
    Network6: 0.0625
    Network7: 0
    Network8: 0
k)Bound approx --> True, num_generators=1000: singh stepsize = (pi / 32) , singh
    Network1: 0.1875
    Network2: 0.2188
    Network3: 0.9375
    Network4: 0.0625
    Network5: 0.3438
    Network6: 0.3125
    Network7: 0
    Network8: 0.0156
l)Bound approx --> True, num_generators=1000: singh stepsize = (pi / 64) , singh
    Network1: 0.6016
    Network2: 0.7656
    Network3: 1
    Network4: 0.2188
    Network5: 0.6250
    Network6: 1
    Network7: 0
    Network8: 0.2500
m)Bound approx --> True, num_generators=1000: singh stepsize = (pi / 64) , singh, dt = 0.01
    Network1: 0.5859
    Network2: 0.7578
    Network3: 1
    Network4: 0.1406
    Network5: 0.6328 --> This is weird --> TODO: Check if error occured
    Network6: 1
    Network7: 0
    Network8: 0.2422
n)Bound approx --> True, num_generators=1000: singh stepsize = (pi / 32) , singh, dt = 0.01
    Network1: 0.1250
    Network2: 0.2188
    Network3: 0.9375
    Network4: 0.0312
    Network5: 0.3438
    Network6: 0.3125
    Network7: 0
    Network8: 0.0156
o)Bound approx --> True, num_generators=1000: singh stepsize = (pi / 16) , regression, dt = 0.01
    Network1: 0
    Network2: 0
    Network3: 0.2812
    Network4: 0
    Network5: 0.1875
    Network6: 0.0625
    Network7: 0
    Network8: 0



p)Bound approx --> True, num_generators=1000: regression stepsize = (pi / 16) , regression, dt = 0.01
    Network1: 0.7812
    Network2: 0.7500
    Network3: 0.9688
    Network4: 0
    Network5: 0.6562
    Network6: 1
    Network7: 0
    Network8: 0.2500


TODO: Play with maxerror