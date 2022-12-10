import torch
from meta_opt import *




def generate():
    a  = torch.tensor([1.0,2.0,3.0], requires_grad=True)
    o = chain_meta_optimizers([a], [SGD, SGD], [[1.0], [0.01]])
    return a, o
def test(a, o):
    log(f"a grad: {a.grad}")
    b = torch.sum(a*a)
    b.backward()
    o.step()
    o.zero_grad()
    return a,o



def  test_triple_sgd():
    a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

    o = chain_meta_optimizers([a], [SGD, SGD, SGD], [[2.0], [0.1], [0.01]])

    def _take_step():
        o.zero_grad()
        l = 0.5 * torch.sum(a*a)
        l.backward()
        o.step()

    def assert_state(expected_a, expected_lr1, expected_lr2, expected_lr3):
        expected_a = torch.tensor(expected_a, dtype=a.dtype)
        expected_lr1 = torch.tensor(expected_lr1, dtype=a.dtype)
        expected_lr2 = torch.tensor(expected_lr2, dtype=a.dtype)
        expected_lr3 = torch.tensor(expected_lr3, dtype=a.dtype)

        assert torch.allclose(a, expected_a), f"check failed: a {a} not close to {expected_a}"

        lr1 = o.lr
        lr2 = o.upper_opt.lr
        lr3 = o.upper_opt.upper_opt.lr

        assert torch.allclose(lr3, expected_lr3), f"check failed: lr3 {lr3} not close to {expected_lr3}"
        assert torch.allclose(lr2, expected_lr2), f"check failed: lr2 {lr2} not close to {expected_lr2}"
        assert torch.allclose(lr1, expected_lr1), f"check failed: lr1 {lr1} not close to {expected_lr1}"
    
    def cycle_vals(g_vals, g):
        g_vals.insert(0, g)
        g_vals.pop()

    # current state:
    # a = [1.0, 2.0, 3.0]
    # first lr = 2.0
    # second lr = 0.1
    # third lr = 0.01

    g_vals = [torch.zeros_like(a), torch.zeros_like(a), torch.zeros_like(a)]


    _take_step()
    
    cycle_vals(g_vals, a.grad)
    # grad_a = [1.0, 2.0, 3.0]
    # next a = [-1, -2, -3]
    # lrs have not updated yet
    # print("wat")
    assert_state(
        [-1.0, -2.0, -3.0],
        2.0,
        0.1,
        0.01
    )

    
    _take_step()
    # grad_a = [-1.0, -2.0, -3.0]
    # <g , gprev> = -14



    # a <- a - gprev*lr1
    # lr1 <- lr1 + <g, gprev> lr2
    # lr2 <- lr2 + <g, gprev>*<gprev, gprevprev> lr3
    

    assert_state(
        [1.0, 2.0, 3.0],
        0.6,
        0.1,
        0.01
    )

    _take_step()
    # grad_a = [1.0, 2.0, 3.0]

    # <g , gprev> = -14

    # <g, gprev>* <gprev, gprevprev> = 196

    assert_state(
        [0.4, 0.8, 1.2],
        -0.8,
        2.06,
        0.01
    )

    _take_step()
    # grad_a = [0.4, 0.8, 1.2]

    # <g , gprev> = 5.6

    # <g, gprev>* <gprev, gprevprev> = -78.4



    assert_state(
        [0.72, 1.44, 2.16],
        10.736,
        1.276,
        0.01
    )



def  test_reverse_triple_sgd():
    global LOGGING_ON 
    a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

    o = reverse_chain_meta_optimizers([a], [SGD, SGD, SGD], [[2.0], [0.1], [0.01]], [{}, {}, {}])

    def _take_step():
        o.zero_grad()
        l = 0.5 * torch.sum(a*a)
        l.backward()
        o.step()

    def assert_state(expected_a, expected_lr1, expected_lr2, expected_lr3):
        expected_a = torch.tensor(expected_a, dtype=a.dtype)
        expected_lr1 = torch.tensor(expected_lr1, dtype=a.dtype)
        expected_lr2 = torch.tensor(expected_lr2, dtype=a.dtype)
        expected_lr3 = torch.tensor(expected_lr3, dtype=a.dtype)


        lr3 = o.lr
        lr2 = o.lower_opt.lr
        lr1 = o.lower_opt.lower_opt.lr

        assert torch.allclose(lr3, expected_lr3), f"check failed: lr3 {lr3} not close to {expected_lr3}"
        assert torch.allclose(lr2, expected_lr2), f"check failed: lr2 {lr2} not close to {expected_lr2}"
        assert torch.allclose(lr1, expected_lr1), f"check failed: lr1 {lr1} not close to {expected_lr1}"

        assert torch.allclose(a, expected_a), f"check failed: a {a} not close to {expected_a}"
    
    def cycle_vals(g_vals, g):
        g_vals.insert(0, g)
        g_vals.pop()

    # current state:
    # a = [1.0, 2.0, 3.0]
    # first lr = 2.0
    # second lr = 0.1
    # third lr = 0.01

    g_vals = [torch.zeros_like(a), torch.zeros_like(a), torch.zeros_like(a)]


    _take_step()
    
    cycle_vals(g_vals, a.grad)
    # grad_a = [1.0, 2.0, 3.0]
    # next a = [-1, -2, -3]
    # lrs have not updated yet
    # print("wat")
    assert_state(
        [-1.0, -2.0, -3.0],
        2.0,
        0.1,
        0.01
    )

    
    _take_step()
    # grad_a = [-1.0, -2.0, -3.0]
    # <g , gprev> = -14



    # a <- a - g * (lr1 - metag * (lr2 - metametag * lr3))
    # metag = -<g, gprev>
    # metametag = <g, gprev> * metagprev

    # g = [-1, -2, -3]
    # metag = 14
    # metametag = 0


    # a <- a - gprev*(lr1 - metag * lr2)
    # lr1 <- lr1 + <g, gprev> lr2
    # lr2 <- lr2 + <g, gprev>*<gprev, gprevprev> lr3
    


    assert_state(
        [-0.4, -0.8, -1.2],
        0.6,
        0.1,
        0.01
    )

    _take_step()



    # a <- a - g * (lr1 - metag * (lr2 - metametag * lr3))
    # metag = -<g, gprev>
    # metametag = <g, gprev> * metagprev

    # g = [-0.4, -0.8, -1.2]
    # metag = 5.6
    # metametag = 78.4


    assert_state(
        [-1.69216, -3.38432, -5.07648],
        -3.2304,
        -0.684,
        0.01
    )


    LOGGING_ON = True
    _take_step()

    # a <- a - g * (lr1 - metag * (lr2 - metametag * lr3))
    # metag = -<g, gprev>
    # metametag = <g, gprev> * metagprev

    # g = [-1.69216, -3.38432, -5.07648]
    # metag = -9.476096
    # metametag = -53.0661376

    # grad_a = [0.4, 0.8, 1.2]

    # <g , gprev> = 5.6

    # <g, gprev>* <gprev, gprevprev> = -78.4



    assert_state(
        [ -9.61730933, -19.23461865, -28.85192798],
        -4.683451521531905,
        -0.153338624,
        0.01
    )





def  test_reverse_positive_triple_sgd():
    global LOGGING_ON 
    a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

    o = reverse_chain_meta_optimizers([a], [SGD, SGD, SGD], [[2.0], [0.1], [0.01]], [{}, {'min_bound': 0.0}, {'min_bound': 0.0}])

    def _take_step():
        o.zero_grad()
        l = 0.5 * torch.sum(a*a)
        l.backward()
        o.step()

    def assert_state(expected_a, expected_lr1, expected_lr2, expected_lr3):
        expected_a = torch.tensor(expected_a, dtype=a.dtype)
        expected_lr1 = torch.tensor(expected_lr1, dtype=a.dtype)
        expected_lr2 = torch.tensor(expected_lr2, dtype=a.dtype)
        expected_lr3 = torch.tensor(expected_lr3, dtype=a.dtype)


        lr3 = o.lr
        lr2 = o.lower_opt.lr
        lr1 = o.lower_opt.lower_opt.lr

        assert torch.allclose(lr3, expected_lr3), f"check failed: lr3 {lr3} not close to {expected_lr3}"
        assert torch.allclose(lr2, expected_lr2), f"check failed: lr2 {lr2} not close to {expected_lr2}"
        assert torch.allclose(lr1, expected_lr1), f"check failed: lr1 {lr1} not close to {expected_lr1}"

        assert torch.allclose(a, expected_a), f"check failed: a {a} not close to {expected_a}"
    
    def cycle_vals(g_vals, g):
        g_vals.insert(0, g)
        g_vals.pop()

    # current state:
    # a = [1.0, 2.0, 3.0]
    # first lr = 2.0
    # second lr = 0.1
    # third lr = 0.01

    g_vals = [torch.zeros_like(a), torch.zeros_like(a), torch.zeros_like(a)]


    _take_step()
    
    cycle_vals(g_vals, a.grad)
    # grad_a = [1.0, 2.0, 3.0]
    # next a = [-1, -2, -3]
    # lrs have not updated yet
    # print("wat")
    assert_state(
        [-1.0, -2.0, -3.0],
        2.0,
        0.1,
        0.01
    )

    
    _take_step()
    # grad_a = [-1.0, -2.0, -3.0]
    # <g , gprev> = -14



    # a <- a - g * (lr1 - metag * (lr2 - metametag * lr3))
    # metag = -<g, gprev>
    # metametag = <g, gprev> * metagprev

    # g = [-1, -2, -3]
    # metag = 14
    # metametag = 0


    # a <- a - gprev*(lr1 - metag * lr2)
    # lr1 <- lr1 + <g, gprev> lr2
    # lr2 <- lr2 + <g, gprev>*<gprev, gprevprev> lr3
    


    assert_state(
        [-0.4, -0.8, -1.2],
        0.6,
        0.1,
        0.01
    )

    _take_step()



    # a <- a - g * (lr1 - metag * (lr2 - metametag * lr3))
    # metag = -<g, gprev>
    # metametag = <g, gprev> * metagprev

    # g = [-0.4, -0.8, -1.2]
    # metag = 5.6
    # metametag = 78.4


    assert_state(
        [-0.16, -0.32, -0.48],
        0.6,
        0.0,
        0.01
    )


    # LOGGING_ON = True
    _take_step()

    # # a <- a - g * (lr1 - metag * (lr2 - metametag * lr3 ))
    # # metag = -<g, gprev> 
    # # metametag = <g, gprev> * metagprev 

    # # g = [-0.16, -0.32, -0.48]
    # # metag = -0.896
    # # metametag = -5.0176

    # # grad_a = [0.4, 0.8, 1.2]

    # # <g , gprev> = 5.6

    # # <g, gprev>* <gprev, gprevprev> = -78.4



    assert_state(
        [-0.05680677, -0.11361354, -0.17042031],
        0.644957696,
        0.050176,
        0.01
    )