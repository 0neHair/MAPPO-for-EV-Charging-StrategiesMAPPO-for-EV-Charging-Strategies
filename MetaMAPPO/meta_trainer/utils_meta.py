import torch

def update_module_params(module: torch.nn.Module, new_params: dict):
    def update(module: torch.nn.Module, name, new_param):
        del module._parameters[name]
        setattr(module, name, new_param)
        module._parameters[name] = new_param

    named_module_dict = dict(module.named_modules())
    for name, new_param, in new_params.items():
        if "." in name:
            module_name, param_name = name.rsplit(".", 1)  # policy.0.bias -> module_name: policy.0, param_name: bias
            update(named_module_dict[module_name], param_name, new_param)
        else:
            update(module, name, new_param)

@torch.no_grad()
def choose_cation(
        args, e, agent, 
        obs_n, share_obs,
        best=False, valid=False, task_id=0
    ):
    if best:
        caction, clog_prob = agent.select_best_caction(obs_n)
    else:
        caction, clog_prob = agent.select_caction(obs_n)
    if not args.ctde:
        if valid:
            agent.validBuffer[task_id].cpush_last_state(state=obs_n, share_state=obs_n, action=caction, log_prob=clog_prob, env_id=e)
        else:
            agent.rolloutBuffer[task_id].cpush_last_state(state=obs_n, share_state=obs_n, action=caction, log_prob=clog_prob, env_id=e)
    else:
        if valid:
            agent.validBuffer[task_id].cpush_last_state(state=obs_n, share_state=share_obs, action=caction, log_prob=clog_prob,env_id=e)
        else:
            agent.rolloutBuffer[task_id].cpush_last_state(state=obs_n, share_state=share_obs, action=caction, log_prob=clog_prob,env_id=e)
    return caction[0]

@torch.no_grad()
def choose_raction(
        args, e, agent, 
        obs_feature_n, obs_n, obs_mask_n,
        global_cs_feature, share_obs,
        best=False, valid=False, task_id=0
    ):
    if args.mode in ['GH']:
        if best:
            raction, rlog_prob = agent.select_best_raction(obs_feature_n, obs_n, obs_mask_n)
        else:
            raction, rlog_prob = agent.select_raction(obs_feature_n, obs_n, obs_mask_n)
    else:
        if best:
            raction, rlog_prob = agent.select_best_raction(obs_n, obs_mask_n) 
        else:
            raction, rlog_prob = agent.select_raction(obs_n, obs_mask_n) 
    if not args.ctde:
        if valid:
            agent.validBuffer[task_id].rpush_last_state(
                    obs_feature=obs_feature_n, 
                    global_cs_feature=obs_feature_n,
                    rstate=obs_n, 
                    share_rstate=obs_n, 
                    action=raction, 
                    action_mask=obs_mask_n, 
                    log_prob=rlog_prob,
                    env_id=e
                )
        else:
            agent.rolloutBuffer[task_id].rpush_last_state(
                    obs_feature=obs_feature_n, 
                    global_cs_feature=obs_feature_n,
                    rstate=obs_n, 
                    share_rstate=obs_n, 
                    action=raction, 
                    action_mask=obs_mask_n, 
                    log_prob=rlog_prob,
                    env_id=e
                )
    else:
        if valid:
            agent.validBuffer[task_id].rpush_last_state(
                    obs_feature=obs_feature_n, 
                    global_cs_feature=global_cs_feature,
                    rstate=obs_n, 
                    share_rstate=share_obs, 
                    action=raction, 
                    action_mask=obs_mask_n, 
                    log_prob=rlog_prob,
                    env_id=e
                )
        else:
            agent.rolloutBuffer[task_id].rpush_last_state(
                    obs_feature=obs_feature_n, 
                    global_cs_feature=global_cs_feature,
                    rstate=obs_n, 
                    share_rstate=share_obs, 
                    action=raction, 
                    action_mask=obs_mask_n, 
                    log_prob=rlog_prob,
                    env_id=e
                )
    return raction[0]

def push_last_c(
        args, e, agent,
        creward_n, 
        obs_n,
        share_obs,
        done_n, valid=False, task_id=0
):
    if not args.ctde:
        if valid:
            agent.validBuffer[task_id].cpush(
                reward=creward_n,
                next_state=obs_n, 
                next_share_state=obs_n, 
                done=done_n,
                env_id=e
            )
        else:
            agent.rolloutBuffer[task_id].cpush(
                reward=creward_n,
                next_state=obs_n, 
                next_share_state=obs_n, 
                done=done_n,
                env_id=e
            )
    else:
        if valid:
            agent.validBuffer[task_id].cpush(
                    reward=creward_n,
                    next_state=obs_n, 
                    next_share_state=share_obs, 
                    done=done_n,
                    env_id=e
                )
        else:
            agent.rolloutBuffer[task_id].cpush(
                    reward=creward_n,
                    next_state=obs_n, 
                    next_share_state=share_obs, 
                    done=done_n,
                    env_id=e
                )

def push_last_r(
        args, e, agent,
        rreward_n, 
        obs_feature_n, obs_n,
        global_cs_feature, share_obs, action_mask,
        done_n, valid=False, task_id=0
):
    if not args.ctde:
        if valid:
            agent.validBuffer[task_id].rpush(
                    reward=rreward_n,
                    next_obs_feature=obs_feature_n, 
                    next_global_cs_feature=obs_feature_n,
                    next_rstate=obs_n, 
                    next_share_rstate=obs_n, 
                    next_action_mask=action_mask,
                    done=done_n,
                    env_id=e
                )
        else:
            agent.rolloutBuffer[task_id].rpush(
                    reward=rreward_n,
                    next_obs_feature=obs_feature_n, 
                    next_global_cs_feature=obs_feature_n,
                    next_rstate=obs_n, 
                    next_share_rstate=obs_n, 
                    next_action_mask=action_mask,
                    done=done_n,
                    env_id=e
                )
    else:
        if valid:
            agent.validBuffer[task_id].rpush(
                    reward=rreward_n,
                    next_obs_feature=obs_feature_n, 
                    next_global_cs_feature=global_cs_feature,
                    next_rstate=obs_n, 
                    next_share_rstate=share_obs, 
                    next_action_mask=action_mask,
                    done=done_n,
                    env_id=e
                )
        else:
            agent.rolloutBuffer[task_id].rpush(
                    reward=rreward_n,
                    next_obs_feature=obs_feature_n, 
                    next_global_cs_feature=global_cs_feature,
                    next_rstate=obs_n, 
                    next_share_rstate=share_obs, 
                    next_action_mask=action_mask,
                    done=done_n,
                    env_id=e
                )