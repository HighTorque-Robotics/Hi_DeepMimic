import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls

import numpy as onp

import pyroki as pk

from typing import Sequence

def solve_retargeting(
    robot: pk.Robot,
    coll: pk.collision.RobotCollision,

    target_link_names: Sequence[str],
    foot_link_names: Sequence[str],

    target_positions: onp.ndarray,
    target_wxyzs: onp.ndarray,
    prev_pos: onp.ndarray, # 为了平滑
    prev_wxyz: onp.ndarray, # 为了平滑
    prev_foot_pos: onp.ndarray, # 为了不打滑
    prev_foot_wxyz: onp.ndarray, # 为了不打滑
    prev_cfg: onp.ndarray,

    link_pos_weight: onp.ndarray = onp.array(5.0),
    link_ori_weight: onp.ndarray = onp.array(0.0),
    fix_base_position: tuple[bool, bool, bool] = [False, False, False],
    fix_base_orientation: tuple[bool, bool, bool] = [False, False, False],
) -> tuple[onp.ndarray, onp.ndarray, onp.ndarray]:
    num_target = len(target_link_names)
    num_foot = len(foot_link_names)

    assert target_positions.shape == (num_target, 3)
    assert target_wxyzs.shape == (num_target, 4)
    assert prev_pos.shape == (3,) and prev_wxyz.shape == (4,)
    assert prev_cfg.shape == (robot.joints.num_actuated_joints,)
    assert prev_foot_pos.shape == (num_foot, 3)
    assert prev_foot_wxyz.shape == (num_foot, 4)
    target_link_indices = [robot.links.names.index(name) for name in target_link_names]
    foot_link_indices = [robot.links.names.index(name) for name in foot_link_names]

    T_world_targets = jaxlie.SE3(
        jnp.concatenate([jnp.array(target_wxyzs), jnp.array(target_positions)], axis=-1) 
    )
    T_world_prev_foot = jaxlie.SE3(
        jnp.concatenate([jnp.array(prev_foot_wxyz), jnp.array(prev_foot_pos)], axis=-1)
    )
    base_pose, cfg = _solve_ik_jax(
        robot,
        coll,
        
        T_world_targets,
        jnp.array(target_link_indices),

        jnp.array(fix_base_position + fix_base_orientation),
        jnp.array(prev_pos),
        jnp.array(prev_wxyz),
        jnp.array(prev_cfg),

        T_world_prev_foot,
        jnp.array(foot_link_indices),

        link_pos_weight,
        link_ori_weight,
    )
    assert cfg.shape == (robot.joints.num_actuated_joints,)

    base_pos = base_pose.translation()
    base_wxyz = base_pose.rotation().wxyz
    assert base_pos.shape == (3,) and base_wxyz.shape == (4,)

    return onp.array(base_pos), onp.array(base_wxyz), onp.array(cfg)

def fk_my(robot: pk.Robot,
       base_pos: onp.ndarray,
       base_wxyz: onp.ndarray,
       cfg: onp.ndarray, 
       link_names: Sequence[str]):
    assert base_pos.shape == (3, )
    assert base_wxyz.shape == (4, )
    assert cfg.shape == (robot.joints.num_actuated_joints,)
    link_indices = [robot.links.names.index(name) for name in link_names]
    T_world_root = jaxlie.SE3(
        jnp.concatenate([jnp.array(base_wxyz), \
                         jnp.array(base_pos)], axis=-1) 
    )
    T_base_all = robot.forward_kinematics(cfg)
    T_base_target = jaxlie.SE3(T_base_all[link_indices, :])
    T_world_target = T_world_root @ T_base_target
    target_pos = onp.array(T_world_target.translation())
    target_wxyzs = onp.array(T_world_target.rotation().wxyz)
    assert target_pos.shape == (len(link_names), 3)
    assert target_wxyzs.shape == (len(link_names), 4)
    return onp.array(target_pos), onp.array(target_wxyzs)

@jdc.jit
def _solve_ik_jax(
    robot: pk.Robot,
    coll: pk.collision.RobotCollision,

    T_world_target: jaxlie.SE3,
    target_joint_indices: jnp.ndarray,
    
    fix_base: jnp.ndarray,
    prev_pos: jnp.ndarray,
    prev_wxyz: jnp.ndarray,
    prev_cfg: jnp.ndarray,

    T_world_prev_foot: jaxlie.SE3,
    foot_link_indices: jnp.ndarray,

    pos_weight: jnp.ndarray = jnp.array(5.0),
    ori_weight: jnp.ndarray = jnp.array(0.0),
) -> tuple[jaxlie.SE3, jax.Array]:
    joint_var = robot.joint_var_cls(0)

    def retract_fn(transform: jaxlie.SE3, delta: jax.Array) -> jaxlie.SE3:
        """Same as jaxls.SE3Var.retract_fn, but removing updates on certain axes."""
        delta = delta * (1 - fix_base)
        return jaxls.SE3Var.retract_fn(transform, delta)

    class ConstrainedSE3Var(
        jaxls.Var[jaxlie.SE3],
        default_factory=lambda: jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3(prev_wxyz),
            prev_pos,
        ),
        tangent_dim=jaxlie.SE3.tangent_dim,
        retract_fn=retract_fn,
    ): ...
    base_var = ConstrainedSE3Var(0)
    # import ipdb; ipdb.set_trace();
    factors = [
        pk.costs.pose_cost_with_base( # 关键点跟踪
            robot,
            joint_var,
            base_var,
            T_world_target,
            target_joint_indices,
            pos_weight=pos_weight,
            ori_weight=ori_weight,
        ),
        pk.costs.limit_cost( # 关节限位
            robot,
            joint_var,
            jnp.array(100.0),
        ),
        pk.costs.rest_with_base_cost( # 与默认值的偏离
            joint_var,
            base_var,
            jnp.array(joint_var.default_factory()),
            jnp.array(
                [0.01] * robot.joints.num_actuated_joints
                + [0.1] * 3  # Base position DoF.
                + [0.001] * 3,  # Base orientation DoF.
            ),
        ),
        # pk.costs.pose_cost_with_base( # 接触脚不动 
        #     robot,
        #     joint_var,
        #     base_var,
        #     T_world_prev_foot,
        #     foot_link_indices,
        #     pos_weight = prev_foot_pos_weight,
        #     ori_weight = prev_foot_wxyz_weight,
        # ),

        pk.costs.foot_clearance_cost(  
            robot,
            joint_var,
            base_var,
            foot_link_indices,
            T_world_prev_foot,
            height_weight = 50.0,
            slip_weight = 2.0
        ),

        # pk.costs.self_collision_cost( # 自碰撞
        #     robot,
        #     robot_coll=coll,
        #     joint_var=joint_var,
        #     margin=0.01,
        #     weight=5.0,
        # ),
        pk.costs.my_smoothness_cost(
            curr_joint_var = joint_var,
            prev_joint = prev_cfg,
            weight = 1.0,
        )
    ]
    sol = (
        jaxls.LeastSquaresProblem(factors, [joint_var, base_var])
        .analyze()
        .solve(
            initial_vals=jaxls.VarValues.make(
                [joint_var.with_value(prev_cfg), base_var]
            ),
            verbose=False,
        )
    )

    return sol[base_var], sol[joint_var]