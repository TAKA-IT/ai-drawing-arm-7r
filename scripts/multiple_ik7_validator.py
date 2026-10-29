from pathlib import Path
#!/usr/bin/env python3
import importlib.util, time
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

FK=str(Path(__file__).resolve().parent / 'fk7_validator.py')
JAC=str(Path(__file__).resolve().parent / 'jacobian7_validator.py')
NAMES=[f'joint_{i}' for i in range(1,8)]
LO=np.array([-3.14159,-2.61799,-2.61799,-3.14159,-2.61799,-3.14159,-3.14159])
HI=np.array([3.14159,2.61799,2.61799,3.14159,2.61799,3.14159,3.14159])

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise RuntimeError('load failed: '+path)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def rv(R):
    c=float(np.clip((np.trace(R)-1)/2,-1,1)); a=float(np.arccos(c))
    if a<1e-10: return np.zeros(3)
    return a*np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])/(2*np.sin(a))

def err(target,current):
    return np.r_[target[:3,3]-current[:3,3],rv(target[:3,:3]@current[:3,:3].T)]

def solve(fkm,jm,chain,target,seed,reference):
    q=np.clip(seed.copy(),LO,HI)
    for k in range(1200):
        T=fkm.calculate_fk(chain,dict(zip(NAMES,q))); e=err(target,T)
        if np.linalg.norm(e[:3])<1e-5 and np.linalg.norm(e[3:])<1e-5: return q,k+1
        J=jm.numerical_jacobian(fkm,chain,q,1e-6); s=np.linalg.svd(J,compute_uv=False); sigma=float(s[-1])
        lam=0.002 if sigma>=0.02 else 0.002+0.028*(1-sigma/0.02)**2
        pinv=J.T@np.linalg.solve(J@J.T+lam*lam*np.eye(6),np.eye(6))
        dq=pinv@e + 0.04*(np.eye(7)-pinv@J)@(reference-q)
        m=float(np.max(np.abs(dq)))
        if m>0.08: dq*=0.08/m
        base=float(np.linalg.norm(e)); accepted=False
        for alpha in [1,0.5,0.25,0.125,0.0625,0.03125,0.015625,0.0078125]:
            candidate=np.clip(q+alpha*dq,LO,HI)
            Tc=fkm.calculate_fk(chain,dict(zip(NAMES,candidate)))
            if np.linalg.norm(err(target,Tc))<base:
                q=candidate; accepted=True; break
        if not accepted: break
    raise RuntimeError('IK did not converge')

class Reader(Node):
    def __init__(self):
        super().__init__('multi_ik_7r'); self.msg=None
        self.create_subscription(JointState,'/joint_states',self.cb,10)
    def cb(self,msg): self.msg=msg

def main():
    rclpy.init(); node=Reader()
    try:
        end=time.monotonic()+10
        while node.msg is None and time.monotonic()<end: rclpy.spin_once(node,timeout_sec=.1)
        if node.msg is None: raise RuntimeError('joint_states timeout')
        state=dict(zip(node.msg.name,node.msg.position)); q0=np.array([state[n] for n in NAMES])
        fkm=load(FK,'fk7_multi'); jm=load(JAC,'jac7_multi'); chain=fkm.load_chain()
        start=fkm.calculate_fk(chain,dict(zip(NAMES,q0))); target=start.copy(); target[:3,3]+=np.array([-0.005,0.010,0.005])
        J0=jm.numerical_jacobian(fkm,chain,q0,1e-6); _,_,vt=np.linalg.svd(J0,full_matrices=True); null=vt[-1]; null/=np.linalg.norm(null)
        ref_a=np.clip(q0+0.45*null,LO,HI); ref_b=np.clip(q0-0.45*null,LO,HI)
        qa,ia=solve(fkm,jm,chain,target,ref_a,ref_a); qb,ib=solve(fkm,jm,chain,target,ref_b,ref_b)
        Ta=fkm.calculate_fk(chain,dict(zip(NAMES,qa))); Tb=fkm.calculate_fk(chain,dict(zip(NAMES,qb)))
        ea=err(target,Ta); eb=err(target,Tb); distance=float(np.linalg.norm(qa-qb))
        np.set_printoptions(precision=9,suppress=True)
        print('solution_a_rad='+np.array2string(qa,separator=', ')); print('solution_b_rad='+np.array2string(qb,separator=', '))
        print('iterations_a={}'.format(ia)); print('iterations_b={}'.format(ib))
        print('solution_distance_rad={:.9f}'.format(distance))
        print('a_position_error_mm={:.9f}'.format(np.linalg.norm(ea[:3])*1000)); print('b_position_error_mm={:.9f}'.format(np.linalg.norm(eb[:3])*1000))
        print('a_orientation_error_deg={:.9f}'.format(np.degrees(np.linalg.norm(ea[3:])))); print('b_orientation_error_deg={:.9f}'.format(np.degrees(np.linalg.norm(eb[3:]))))
        limits=bool(np.all(qa>=LO)&np.all(qa<=HI)&np.all(qb>=LO)&np.all(qb<=HI)); print('joint_limits_respected={}'.format(limits))
        if distance<0.05: raise RuntimeError('solutions not distinct enough')
        if np.linalg.norm(ea[:3])*1000>0.01 or np.linalg.norm(eb[:3])*1000>0.01: raise RuntimeError('position accuracy failed')
        if np.degrees(np.linalg.norm(ea[3:]))>0.001 or np.degrees(np.linalg.norm(eb[3:]))>0.001: raise RuntimeError('orientation accuracy failed')
        if not limits: raise RuntimeError('joint limit violation')
        print('7R_MULTIPLE_IK_SOLUTIONS=PASS')
    finally:
        node.destroy_node(); rclpy.shutdown()
if __name__=='__main__': main()
