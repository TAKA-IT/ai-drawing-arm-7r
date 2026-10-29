from pathlib import Path
#!/usr/bin/env python3
import importlib.util, time
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
FK=str(Path(__file__).resolve().parent / 'fk7_validator.py')
JAC=str(Path(__file__).resolve().parent / 'jacobian7_validator.py')
N=[f'joint_{i}' for i in range(1,8)]
LO=np.array([-3.14159,-2.61799,-2.61799,-3.14159,-2.61799,-3.14159,-3.14159])
HI=np.array([3.14159,2.61799,2.61799,3.14159,2.61799,3.14159,3.14159])
def load(path,name):
 s=importlib.util.spec_from_file_location(name,path)
 if s is None or s.loader is None: raise RuntimeError('load failed')
 m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def rv(R):
 c=float(np.clip((np.trace(R)-1)/2,-1,1)); a=float(np.arccos(c))
 if a<1e-10:return np.zeros(3)
 return a*np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])/(2*np.sin(a))
def error(target,current):return np.r_[target[:3,3]-current[:3,3],rv(target[:3,:3]@current[:3,:3].T)]
def solve(fkm,jm,chain,target,seed):
 q=seed.copy()
 for k in range(500):
  T=fkm.calculate_fk(chain,dict(zip(N,q))); e=error(target,T)
  if np.linalg.norm(e[:3])<1e-5 and np.linalg.norm(e[3:])<1e-5:return q,k+1
  J=jm.numerical_jacobian(fkm,chain,q,1e-6); sigma=float(np.linalg.svd(J,compute_uv=False)[-1])
  lam=.002 if sigma>=.02 else .002+.028*(1-sigma/.02)**2
  dq=J.T@np.linalg.solve(J@J.T+lam*lam*np.eye(6),e); m=float(np.max(np.abs(dq)))
  if m>.06:dq*=.06/m
  base=float(np.linalg.norm(e)); accepted=False
  for a in [1,.5,.25,.125,.0625,.03125,.015625,.0078125]:
   cand=np.clip(q+a*dq,LO,HI); Tc=fkm.calculate_fk(chain,dict(zip(N,cand)))
   if np.linalg.norm(error(target,Tc))<base:q=cand;accepted=True;break
  if not accepted:break
 raise RuntimeError('point IK failed')
class Reader(Node):
 def __init__(self):
  super().__init__('circle_ik_7r');self.msg=None;self.create_subscription(JointState,'/joint_states',self.cb,10)
 def cb(self,m):self.msg=m
def main():
 rclpy.init();node=Reader()
 try:
  end=time.monotonic()+10
  while node.msg is None and time.monotonic()<end:rclpy.spin_once(node,timeout_sec=.1)
  if node.msg is None:raise RuntimeError('joint_states timeout')
  state=dict(zip(node.msg.name,node.msg.position));q0=np.array([state[n] for n in N])
  fkm=load(FK,'fkc');jm=load(JAC,'jacc');chain=fkm.load_chain();start=fkm.calculate_fk(chain,dict(zip(N,q0)))
  center=start[:3,3]+np.array([-.035,0,0]);radius=.025;points=73;q=q0.copy();qs=[];actual=[];ori=[];iters=[]
  for i,theta in enumerate(np.linspace(0,2*np.pi,points)):
   target=start.copy();target[:3,3]=center+np.array([0,radius*np.cos(theta),radius*np.sin(theta)])
   q,k=solve(fkm,jm,chain,target,q);T=fkm.calculate_fk(chain,dict(zip(N,q)))
   qs.append(q.copy());actual.append(T[:3,3].copy());ori.append(np.degrees(np.linalg.norm(rv(start[:3,:3]@T[:3,:3].T))));iters.append(k)
  qs=np.array(qs);actual=np.array(actual);rho=np.sqrt((actual[:,1]-center[1])**2+(actual[:,2]-center[2])**2)
  rmse=float(np.sqrt(np.mean((rho-radius)**2))*1000);maxori=float(np.max(ori));closure=float(np.linalg.norm(actual[-1]-actual[0])*1000)
  jumps=np.linalg.norm(np.diff(qs,axis=0),axis=1);maxjump=float(np.max(jumps));limits=bool(np.all(qs>=LO)&np.all(qs<=HI))
  np.savez('/tmp/circle7_trajectory.npz',joint_names=np.array(N),positions=qs,actual_tcp=actual,center=center,radius=radius)
  print(f'point_count={points}');print(f'radius_mm={radius*1000:.6f}');print(f'radius_rmse_mm={rmse:.9f}');print(f'max_orientation_error_deg={maxori:.9f}');print(f'closure_error_mm={closure:.9f}');print(f'max_joint_step_rad={maxjump:.9f}');print(f'max_ik_iterations={max(iters)}');print(f'joint_limits_respected={limits}')
  if rmse>.02:raise RuntimeError('radius RMSE failed')
  if maxori>.001:raise RuntimeError('orientation failed')
  if closure>.02:raise RuntimeError('closure failed')
  if maxjump>.15:raise RuntimeError('joint continuity failed')
  if not limits:raise RuntimeError('limits failed')
  print('7R_TCP_CIRCLE_OFFLINE=PASS')
 finally:node.destroy_node();rclpy.shutdown()
if __name__=='__main__':main()
