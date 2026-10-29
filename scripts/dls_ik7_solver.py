from pathlib import Path
#!/usr/bin/env python3
import importlib.util
import time
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

FK_PATH=str(Path(__file__).resolve().parent / 'fk7_validator.py')
JAC_PATH=str(Path(__file__).resolve().parent / 'jacobian7_validator.py')
JOINTS=[f'joint_{i}' for i in range(1,8)]
LOW=np.array([-3.14159,-2.61799,-2.61799,-3.14159,-2.61799,-3.14159,-3.14159])
HIGH=np.array([3.14159,2.61799,2.61799,3.14159,2.61799,3.14159,3.14159])

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise RuntimeError(f'cannot load {path}')
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def rotvec(R):
    c=float(np.clip((np.trace(R)-1)/2,-1,1)); a=float(np.arccos(c))
    if a<1e-10: return np.zeros(3)
    return a*np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])/(2*np.sin(a))

def pose_error(target,current):
    return np.r_[target[:3,3]-current[:3,3],rotvec(target[:3,:3]@current[:3,:3].T)]

class Reader(Node):
    def __init__(self):
        super().__init__('dls_ik_validator_7r'); self.msg=None
        self.create_subscription(JointState,'/joint_states',self.cb,10)
    def cb(self,msg): self.msg=msg

def main():
    rclpy.init(); node=Reader()
    try:
        end=time.monotonic()+10
        while node.msg is None and time.monotonic()<end: rclpy.spin_once(node,timeout_sec=.1)
        if node.msg is None: raise RuntimeError('joint_states timeout')
        state=dict(zip(node.msg.name,node.msg.position)); missing=[n for n in JOINTS if n not in state]
        if missing: raise RuntimeError(f'missing joints: {missing}')
        fkmod=load(FK_PATH,'fk7'); jacmod=load(JAC_PATH,'jac7'); chain=fkmod.load_chain()
        q0=np.array([state[n] for n in JOINTS]); q=q0.copy(); start=fkmod.calculate_fk(chain,dict(zip(JOINTS,q)))
        target=start.copy(); target[:3,3]+=np.array([-0.005,0.010,0.005])
        converged=False
        for iteration in range(1500):
            current=fkmod.calculate_fk(chain,dict(zip(JOINTS,q))); err=pose_error(target,current)
            pos_norm=float(np.linalg.norm(err[:3])); rot_norm=float(np.linalg.norm(err[3:]))
            if pos_norm<1e-5 and rot_norm<1e-5: converged=True; break
            J=jacmod.numerical_jacobian(fkmod,chain,q,1e-6); s=np.linalg.svd(J,compute_uv=False)
            sigma=float(s[-1]); damping=0.002 if sigma>=0.02 else 0.002+0.028*(1-sigma/0.02)**2
            dq=J.T@np.linalg.solve(J@J.T+(damping*damping)*np.eye(6),err)
            max_abs=float(np.max(np.abs(dq)))
            if max_abs>0.08: dq*=0.08/max_abs
            base=float(np.linalg.norm(err)); accepted=False; alpha=1.0
            while alpha>=1/128:
                candidate=np.clip(q+alpha*dq,LOW,HIGH)
                candidate_tf=fkmod.calculate_fk(chain,dict(zip(JOINTS,candidate)))
                if np.linalg.norm(pose_error(target,candidate_tf))<base:
                    q=candidate; accepted=True; break
                alpha*=0.5
            if not accepted: break
        final=fkmod.calculate_fk(chain,dict(zip(JOINTS,q))); final_err=pose_error(target,final)
        print('iterations={}'.format(iteration+1))
        print('start_tcp_m='+np.array2string(start[:3,3],precision=9,separator=', '))
        print('target_tcp_m='+np.array2string(target[:3,3],precision=9,separator=', '))
        print('solved_tcp_m='+np.array2string(final[:3,3],precision=9,separator=', '))
        print('solution_rad='+np.array2string(q,precision=9,separator=', '))
        print('max_joint_delta_rad={:.9f}'.format(float(np.max(np.abs(q-q0)))))
        print('position_error_mm={:.9f}'.format(float(np.linalg.norm(final_err[:3])*1000)))
        print('orientation_error_deg={:.9f}'.format(float(np.degrees(np.linalg.norm(final_err[3:])))))
        print('joint_limits_respected={}'.format(bool(np.all(q>=LOW)&np.all(q<=HIGH))))
        if not converged: raise RuntimeError('DLS IK did not converge')
        if not (np.all(q>=LOW)&np.all(q<=HIGH)): raise RuntimeError('joint limit violation')
        print('7R_DLS_IK_OFFLINE=PASS')
    finally:
        node.destroy_node(); rclpy.shutdown()
if __name__=='__main__': main()
