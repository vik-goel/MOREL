import os
import time
import joblib
import numpy as np
import os.path as osp
import tensorflow as tf
from baselines import logger
from collections import deque
from baselines.common import explained_variance

from baselines.common import set_global_seeds, explained_variance
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.atari_wrappers import wrap_deepmind
from baselines.common import tf_util

from baselines.a2c.a2c import restore_from_checkpoint, restore_teacher_from_checkpoint

from baselines.a2c.utils import discount_with_dones
from baselines.a2c.utils import Scheduler, make_path, find_trainable_variables
from baselines.a2c.utils import cat_entropy, mse, fc, conv, sample, cat_entropy_softmax, cat_entropy

from baselines.common.cmd_util import publish_attention_weights
import unsupervised_object_segmentation.object_segmentation_network as object_segmentation
import copy

class Model(object):

    def _create_object_segmentation_net(self, X, reuse=False, is_step_model=False, embedding=None):
        assert self.hparams.get('teacher_ckpt')

        scaled_images = tf.cast(X, dtype=tf.float32) / 255.

        # Create a sfm_base.
        frame0 = tf.expand_dims(scaled_images[..., -2], axis=-1)
        frame1 = tf.expand_dims(scaled_images[..., -1], axis=-1)
        print('frame0_shape: {}'.format(frame0.get_shape()))
        print('frame1_shape: {}'.format(frame1.get_shape()))

        sfm_base = object_segmentation.ObjectSegmentationBase(frames=scaled_images[..., -2:], embedding=embedding)

        sfm_hparams = copy.deepcopy(self.hparams)
        sfm_hparams['batch_size'] = int(scaled_images.get_shape()[0])

        result = object_segmentation.ObjectSegmentationNetwork(
            hparams=sfm_hparams,
            sfm_base=sfm_base,
            is_teacher_network=True,
            reuse=reuse,
            is_step_model=is_step_model,
            trainable=self.hparams['do_joint_training']
        )

        return result


    def _object_masks_to_attention(self, object_masks, object_translation):
        use_mask = tf.cast(tf.reduce_sum(object_masks, axis=[1, 2]) > 2.0, tf.float32)
        teach_mask = tf.cast(tf.reduce_sum(use_mask, axis=-1) > 0, tf.float32)
        object_masks *= tf.expand_dims(tf.expand_dims(use_mask, axis=1), axis=1)
        reduce_op = tf.reduce_max if self.hparams.get('max_attention_truth') else tf.reduce_sum

        flow_norms = tf.reduce_sum(tf.abs(object_translation), axis=-1) + 1e-6
        flow_totals = tf.reduce_sum(flow_norms, axis=-1, keep_dims=True)
        weight = tf.expand_dims(tf.expand_dims(flow_norms / flow_totals, axis=1), axis=1)

        summed_masks = reduce_op(object_masks * weight, axis=-1, keep_dims=True)
        return summed_masks, teach_mask


    def _get_attention_truth(self, teacher, is_step_model):
        # Forward pass the teacher to get object masks.
        # Sum the object masks to get all the object pixels across all the masks.
        summed_masks, teach_mask = self._object_masks_to_attention(teacher.object_masks, teacher.object_translation)

        if not is_step_model:
            tf.summary.image('summed_masks', summed_masks, max_outputs=1)

        if self.hparams.get('correct_attention_truth'):
            coarse_masks = tf.nn.conv2d(summed_masks, filter=np.ones([8,8,1,1],dtype=np.float32), strides=[1,4,4,1], padding='VALID')
            coarse_masks = tf.nn.conv2d(coarse_masks, filter=np.ones([4,4,1,1],dtype=np.float32), strides=[1,2,2,1], padding='VALID')
            coarse_masks = tf.nn.conv2d(coarse_masks, filter=np.ones([3,3,1,1],dtype=np.float32), strides=[1,1,1,1], padding='VALID')
        elif self.hparams.get('resize_attention_downsampling'):
            coarse_masks = tf.image.resize_images(summed_masks, size=[7, 7], method=tf.image.ResizeMethod.AREA)
        else:
            # Make them coarser by summing the entire 12x12 region. (eg. downsample from 84x84 to 7x7 by summing in 12x12 blocks)
            coarse_masks = tf.nn.conv2d(summed_masks, filter=np.ones([12,12,1,1],dtype=np.float32), strides=[1,12,12,1], padding='SAME')

        batch_size = self.nbatch_train

        coarse_masks = tf.reshape(coarse_masks, [batch_size, -1]) + 1e-6
        print('coarse_masks: {}'.format(coarse_masks.get_shape()))

        #attention_truth = tf.nn.softmax(coarse_masks / hparams['attention_truth_temperature'])
        attention_truth = coarse_masks / tf.reduce_sum(coarse_masks, axis=-1, keep_dims=True)

        print('attention_truth: {}'.format(attention_truth.get_shape()))
        # Do not backprop into the labels and try to train the teacher -- the student is not the master.
        attention_truth = tf.stop_gradient(attention_truth)
        return attention_truth, teach_mask


    def _get_attention_truth_20(self, teacher, is_step_model):
        # Forward pass the teacher to get object masks.
        # Sum the object masks to get all the object pixels across all the masks.
        summed_masks, teach_mask = self._object_masks_to_attention(teacher.object_masks, teacher.object_translation)

        if not is_step_model:
            tf.summary.image('summed_masks_20', summed_masks, max_outputs=1)

        #if not is_step_model:
        #    tf.summary.image('summed_masks', summed_masks, max_outputs=1)

        # Make them coarser by summing the entire 12x12 region. (eg. downsample from 84x84 to 7x7 by summing in 12x12 blocks)
        if self.hparams.get('resize_attention_downsampling'):
            coarse_masks = tf.image.resize_images(summed_masks, size=[20, 20], method=tf.image.ResizeMethod.AREA)
        else:
            coarse_masks = tf.nn.conv2d(summed_masks, filter=np.ones([8,8,1,1], dtype=np.float32), strides=[1,4,4,1], padding='VALID')

        batch_size = self.nbatch_train

        coarse_masks = tf.reshape(coarse_masks, [batch_size, -1]) + 1e-6
        print('coarse_masks_20: {}'.format(coarse_masks.get_shape()))

        #attention_truth = tf.nn.softmax(coarse_masks / hparams['attention_truth_temperature'])
        attention_truth = coarse_masks / tf.reduce_sum(coarse_masks, axis=-1, keep_dims=True)
        print('attention_truth_20: {}'.format(attention_truth.get_shape()))


        # Do not backprop into the labels and try to train the teacher -- the student is not the master.
        attention_truth = tf.stop_gradient(attention_truth)
        return attention_truth, teach_mask


    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm, hparams):
        assert hparams != None
        self.hparams = hparams
        self.nbatch_act = nbatch_act
        self.nbatch_train = nbatch_train
        self.nsteps = nsteps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        sess = tf.get_default_session()
        self.sess = sess

        nh, nw, nc = ob_space.shape
        ob_shape_act = (nbatch_act, nh, nw, nc)
        ob_shape_train = (nbatch_train, nh, nw, nc)

        X_act = tf.placeholder(tf.float32, ob_shape_act, name='Ob_step') #obs
        X_train = tf.placeholder(tf.float32, ob_shape_train, name='Ob_train') #obs

        act_model = policy(sess, X_act, ob_space, ac_space, nbatch_act, 1, reuse=False, hparams=hparams)
        train_model = policy(sess, X_train, ob_space, ac_space, nbatch_train, nsteps, reuse=True, hparams=hparams)

        A = train_model.pdtype.sample_placeholder([None])
        ADV = tf.placeholder(tf.float32, [None])
        R = tf.placeholder(tf.float32, [None])
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        OLDVPRED = tf.placeholder(tf.float32, [None])
        LR = tf.placeholder(tf.float32, [])
        CLIPRANGE = tf.placeholder(tf.float32, [])
        TEACHER_C = tf.placeholder(tf.float32, [])

        neglogpac = train_model.pd.neglogp(A)
        entropy = tf.reduce_mean(train_model.pd.entropy())

        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        vf_losses1 = tf.square(vpred - R)
        vf_losses2 = tf.square(vpredclipped - R)
        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))

        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef


        if hparams.get('teacher_ckpt'):
            # assert hparams.get('learn_attention_from_teacher')
            # assert hparams.get('attention_20')

            # We take a max of loss1, loss2. There is no gradient if loss2 > loss1.
            # mask = loss1 < loss2 (eg. 0 if loss2 > loss1)
            train_mask = tf.cast(
                tf.logical_and(
                    vf_losses1 < vf_losses2,
                    pg_losses < pg_losses2),
                tf.float32)

            if hparams['do_joint_training']:
                train_teacher = self._create_object_segmentation_net(train_model.X, reuse=False, embedding=train_model.original_h)
            else:
                train_teacher = self._create_object_segmentation_net(train_model.X, reuse=False)

            # train_attention_truth, train_attention_mask = self._get_attention_truth(train_teacher, is_step_model=False)

            # attention_loss = tf.nn.softmax_cross_entropy_with_logits(
            #     labels=train_attention_truth,
            #     logits=tf.reshape(train_model.attention_logits, [nbatch_train,-1])
            # )
            # print('attention_loss: {}'.format(attention_loss.get_shape()))
            # print('train_attention_mask: {}'.format(train_attention_mask.get_shape()))

            # attention_loss = attention_loss * train_attention_mask
            # attention_loss = tf.reduce_mean(attention_loss)

            # teacher_loss = TEACHER_C * attention_loss
            # loss += teacher_loss

            teacher_loss = 0.0
            tf.summary.scalar('teacher_c', TEACHER_C)

            if hparams['do_joint_training']:
                joint_loss = train_teacher.transform_loss + train_teacher.mask_reg_loss
                print('joint loss: {}'.format(joint_loss))

                joint_loss = joint_loss * train_mask
                print('masked joint loss: {}'.format(joint_loss))

                teacher_loss += tf.reduce_mean(joint_loss)

            if hparams.get('attention_20'):
                truth, mask = self._get_attention_truth_20(train_teacher, is_step_model=False)

                attention_loss_20 = tf.nn.softmax_cross_entropy_with_logits(
                    labels=truth,
                    logits=tf.reshape(train_model.attention_logits_20, [self.nbatch_train, -1])
                )

                # kl = -tf.reduce_sum(OLDNEGLOGATTN20 *
                #     (tf.log(tf.reshape(train_model.attention_weights_20, [-1, 400])) - tf.log(OLDNEGLOGATTN20)),
                #     axis=1)
                #kl = tf.Print(kl, [tf.reduce_mean(kl), tf.reduce_min(kl), tf.reduce_max(kl)])
                #mask *= tf.cast(kl < hparams['attention_epsilon'], tf.float32)

                if hparams['ppo_attention_clip']:
                    mask *= train_mask

                reshaped_logits = tf.reshape(train_model.attention_logits_20, [-1, 400])
                attention_entropy = cat_entropy(reshaped_logits)
                attention_loss_20 -= hparams['attention_entropy_c'] * attention_entropy

                teacher_loss += attention_loss_20
                teacher_loss = tf.reduce_mean(teacher_loss * mask)

                tf.summary.scalar('attention_loss_20', tf.reduce_mean(attention_loss_20))
                teacher_loss += TEACHER_C * teacher_loss

                tf.summary.scalar('attention_entropy', attention_entropy)

            if hparams.get('extrapath_attention_20'):
                print('ATTENTION EXTRAPATH!!!')
                truth, mask = self._get_attention_truth_20(train_teacher, is_step_model=False)

                attention_loss_20 = tf.nn.softmax_cross_entropy_with_logits(
                    labels=truth,
                    logits=tf.reshape(train_model.extrapath_attention_logits_20, [self.nbatch_train, -1])
                )

                if hparams['ppo_attention_clip']:
                    mask *= train_mask

                reshaped_logits = tf.reshape(train_model.extrapath_attention_logits_20, [-1, 400])
                attention_entropy = cat_entropy(reshaped_logits)
                attention_entropy_loss = hparams['attention_entropy_c'] * attention_entropy

                teacher_loss += tf.reduce_mean((- attention_loss_20 - attention_entropy_loss) * mask) * TEACHER_C

                tf.summary.scalar('extrapath_attention_loss_20', tf.reduce_mean(attention_loss_20))
                tf.summary.scalar('extrapath_attention_entropy', attention_entropy)

            loss += teacher_loss
        else:
            assert not hparams.get('learn_attention_from_teacher')

        with tf.variable_scope('model'):
            params = tf.trainable_variables()
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        _train = trainer.apply_gradients(grads)


        steps = 0

        def train(lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None, log_attention=False):
            nonlocal steps

            #teacher_decay_c = hparams['teacher_decay_c']
            #teacher_c = hparams['max_teacher_c'] / (teacher_decay_c * steps + 1)

            # max a step 0
            # 0 at step 1e7

            if not hparams['use_extra_path']:
                lerp = float(steps) / 1e7
                lerp = min(lerp, 1)
                teacher_c = hparams['max_teacher_c'] * (1. - lerp)
            else:
                teacher_c = 1

            # Multiply by 0.25 because we do 4 epochs.
            steps += 0.25 * nbatch_train

            advs = returns - values
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            td_map = {train_model.X:obs, A:actions, ADV:advs, R:returns, LR:lr,
                    CLIPRANGE:cliprange, OLDNEGLOGPAC:neglogpacs, OLDVPRED:values,
                    TEACHER_C:teacher_c}

            if hparams['do_joint_training']:
                td_map[train_teacher.mask_reg_c] = hparams['mask_reg_loss_c']

            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks


            ops = [pg_loss, vf_loss, entropy, approxkl, clipfrac]

            if log_attention:
                ops.append(self.train_model.attention_weights_20)

            ops.append(_train)

            results = sess.run(ops, td_map)
            return results[:-1]

        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

        def save(save_path):
            ps = sess.run(params)
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)
            # If you want to load weights, also save/load observation scaling inside VecNormalize

        self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess) #pylint: disable=E1101

        if hparams.get('teacher_ckpt'):
            # Load in the teacher AFTER doing the init so we don't overwrite the weights.
            restore_teacher_from_checkpoint(sess, hparams['teacher_ckpt'])

class Runner(object):

    def __init__(self, *, env, model, nsteps, gamma, lam):
        self.env = env
        self.model = model
        nenv = env.num_envs
        self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=model.train_model.X.dtype.name)
        self.obs[:] = env.reset()
        self.gamma = gamma
        self.lam = lam
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_states = self.states
        epinfos = []
        for _ in range(self.nsteps):
            actions, values, neglogpacs = self.model.step(self.obs)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, self.states, self.dones)
        #discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
            mb_states, epinfos)
# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

def constfn(val):
    def f(_):
        return val
    return f

def learn(*, policy, env, nsteps, total_timesteps, ent_coef, lr,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=0, hparams=None):
    assert hparams != None


    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    make_model = lambda : Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm, hparams=hparams)
    if save_interval and logger.get_dir():
        import cloudpickle
        with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
            fh.write(cloudpickle.dumps(make_model))
    model = make_model()
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)


    ckpt_path = hparams['restore_from_ckpt_path']
    if ckpt_path:
        restore_from_checkpoint(model.sess, ckpt_path, hparams['first_layer_mode'], hparams)

    epinfobuf = deque(maxlen=100)
    tfirststart = time.time()

    nupdates = total_timesteps//nbatch

    attention_buffer = np.zeros([nsteps, 20, 20, 1])

    # Setup for saving checkpoints.
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=1.0, max_to_keep=3)
    ckpt_dir = os.path.join(hparams['base_dir'], 'ckpts', hparams['experiment_name'])
    try: os.makedirs(ckpt_dir)
    except: pass

    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0
        nbatch_train = nbatch // nminibatches
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = lr(frac)
        cliprangenow = cliprange(frac)
        obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run() #pylint: disable=E0632
        epinfobuf.extend(epinfos)
        mblossvals = []

        if states is None: # nonrecurrent version
            inds = np.arange(nbatch)
            for epoch_index in range(noptepochs):
                log_attention = 'attention' in hparams['policy'] and epoch_index == 0

                np.random.shuffle(inds)
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))

                    vals = model.train(lrnow, cliprangenow, *slices, log_attention=log_attention)
                    if log_attention:
                        attention_val = vals[-1]
                        vals = vals[:-1]

                        for i, idx in enumerate(mbinds):
                            if idx < nsteps:
                                attention_buffer[idx] = attention_val[i]

                    mblossvals.append(vals)

                if log_attention:
                    publish_attention_weights(attention_buffer)

        else: # recurrent version
            assert nenvs % nminibatches == 0
            envsperbatch = nenvs // nminibatches
            envinds = np.arange(nenvs)
            flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
            envsperbatch = nbatch_train // nsteps
            for _ in range(noptepochs):
                np.random.shuffle(envinds)
                for start in range(0, nenvs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mbstates = states[mbenvinds]
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))

        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.time()
        fps = int(nbatch / (tnow - tstart))
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, returns)
            logger.logkv("serial_timesteps", update*nsteps)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update*nbatch)
            logger.logkv("fps", fps)
            logger.logkv("explained_variance", float(ev))
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.logkv('time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv(lossname, lossval)
            logger.dumpkvs()

        # if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
        #     checkdir = osp.join(logger.get_dir(), 'checkpoints')
        #     os.makedirs(checkdir, exist_ok=True)
        #     savepath = osp.join(checkdir, '%.5i'%update)
        #     print('Saving to', savepath)
        #     model.save(savepath)

        if (update-1) % 500 == 0:
            saver.save(model.sess, os.path.join(ckpt_dir, 'my-model'), global_step=update)


        if (update-1) % 1000 == 0:
            try:
                experiment_name = os.path.basename(hparams['base_dir'])
                instance_name = hparams['experiment_name']

                for seperator in ['ckpts/', 'logs/', 'videos/']:
                    if 'video' in seperator:
                        src_instance_name = instance_name + '_rgb'
                    else:
                        src_instance_name = instance_name

                    src_dir = os.path.join(hparams['base_dir'], seperator+src_instance_name)
                    dst_dir = 's3://irl-experiments/{}/{}'.format(experiment_name, seperator+instance_name)

                    command = 'aws s3 sync {} {}'.format(src_dir, dst_dir)
                    print(command)
                    os.system(command)
            except:
                pass
    env.close()

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)
