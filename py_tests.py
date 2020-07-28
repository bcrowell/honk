#!/bin/python3

import math
from pie import Pie

def main():
  assert_equal( Pie.from_string("0 0,1 1").restrict(0.3,0.7)(0.4) , 0.4 )
  assert_equal( Pie.from_string("0 0,1 1,2 2").restrict(0.3,0.7)(0.4) , 0.4 )

def barf(dat):
  raise Exception(' '.join(dat))

def assert_boolean(condition,message):
  if not condition:
    barf([message])

def assert_equal_eps(x,y,eps):
  err = x-y
  if math.isnan(x) or math.isnan(y) or abs(err)>eps:
    barf((["test failed, x=",x,", y=",y,", err=",err,", eps=",eps]))

def assert_rel_equal_eps(x,y,eps):
  if x==0.0 and y==0.0:
    return
  if x==0.0:
    return assert_rel_equal_eps(y,x,eps) # avoid division by zero
  rel_err = (x-y)/x
  if math.isnan(x) or math.isnan(y) or abs(rel_err)>eps:
    barf((["test failed, x=",x,", y=",y,", rel err=",rel_err,", eps=",eps]))

def assert_equal(x,y):
  return assert_equal_eps(x,y,2.0*eps())

def assert_rel_equal(x,y):
  return assert_rel_equal_eps(x,y,2.0*eps())

def assert_rel_equal_eps_vector(x,y,eps):
  for i in range(len(x)):
    assert_rel_equal_eps(x[i],y[i],eps)

def done(verbosity,name):
  if verbosity()>=1:
    print("Passed test_"+name+", language=LANG")

def verbosity():
  return 1

def eps():
  return 1.0e-6

main()
