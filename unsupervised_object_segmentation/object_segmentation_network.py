"""Object segmentation network model definition."""

import numpy as np
import tensorflow as tf
from a2cutils import *

class ObjectSegmentationBase(object):
    def __init__(self, frames, target_frame=None, source_frame=None, actions=None, embedding=None):
        self.frames = frames

        self.target_frame = target_frame
        if self.target_frame == None:
            self.target_frame = self.frames[:, :, :, -2]

        self.source_frame = source_frame
        if self.source_frame == None:
            self.source_frame = self.frames[:, :, :, -1]

        self.actions = actions
        self.embedding = embedding


class ObjectSegmentationNetworkCommon(object):
    def __init__(self, hparams, sfm_base=None, is_teacher_network=False, is_step_model=False, reuse=False, trainable=True):
        self.hparams = hparams
        self.batch_dims = (hparams['batch_size'], hparams['img_h'], hparams['img_w'], hparams['img_c'])
        self.sfm_base = sfm_base
        self.is_teacher_network = is_teacher_network
        self.is_step_model = is_step_model
        self.reuse = reuse
        self.trainable = trainable

    def log_image(self, name, img):
        if not self.is_teacher_network and not self.is_step_model:
            tf.summary.image(name, img, max_outputs=1)

    def log_scalar(self, name, value):
        if not self.is_step_model:
            tf.summary.scalar(name, value)

    def next_learning_rate(self):
        for step in range(self.hparams['batch_size']):
            cur_lr = self.lr_scheduler.value()
        return cur_lr

    def _init_frames_placeholder(self, num_frames):
        with tf.variable_scope('placeholders'):
            self.frames_placeholder = tf.placeholder(
                tf.uint8,
                (self.hparams['batch_size'], self.hparams['img_h'], self.hparams['img_w'], num_frames),
                name='frames_placeholder',
            )
            print(self.frames_placeholder)

        for i in range(self.hparams['num_input_frames']):
            self.log_image('frame{}'.format(i), self.frames_placeholder[:,:,:,i:i+1])

        # Preprocessing.
        with tf.variable_scope('preprocess'):
            self.frames = self._preprocess(self.frames_placeholder)
            # self.target_frame = self._preprocess(self.target_frame_placeholder)
            # self.log_image('diff image', self.target_frame - self.frames[:, :, :, -1:])

    def _preprocess(self, img):
        return tf.cast(img, tf.float32) / 255.

    def _create_train_op(self, loss):
        with tf.variable_scope('train', reuse=self.reuse):
            # Use to decay the learning rate.
            self.learning_rate = tf.placeholder(tf.float32, [])
            self.lr_scheduler = Scheduler(v=self.hparams['learning_rate'], nvalues=self.hparams['total_timesteps'], schedule='linear')

            if self.hparams['optimizer'] == 'adam':
                self.optimizer = tf.train.AdamOptimizer(
                    learning_rate=self.hparams['learning_rate'],
                )
            elif self.hparams['optimizer'] == 'rmsprop':
                self.optimizer = tf.train.RMSPropOptimizer(
                    learning_rate=self.learning_rate,
                    decay=self.hparams['alpha'],
                    epsilon=self.hparams['epsilon'],
                )
            else:
                raise ValueError('unknown optimizer')

            params = find_trainable_variables('model')
            grads = tf.gradients(loss, params)

            # Log the norm of all of the gradients.
            with tf.variable_scope('grad_norms'):
                for grad, param in zip(grads, params):
                    if grad == None: continue

                    norm = tf.norm(grad)
                    self.log_scalar(param.name, norm)

            grads, global_norm = tf.clip_by_global_norm(grads, self.hparams['max_grad_norm'])
            grads = list(zip(grads, params))
            train_op = self.optimizer.apply_gradients(grads)
            self.log_scalar('global_norm', global_norm)
            return train_op


# Code taken from https://github.com/waxz/sfm_net/blob/master/model.ipynb
class ObjectSegmentationNetwork(ObjectSegmentationNetworkCommon):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        scope = 'teacher' if self.is_teacher_network else 'model'

        if not self.sfm_base:
            self._init_inputs()
            self.sfm_base = ObjectSegmentationBase(frames=self.frames)

        with tf.variable_scope(scope):
            self.frames = tf.identity(self.sfm_base.frames, name='frames')
            print(self.frames)

            self.source_frame = tf.expand_dims(self.sfm_base.source_frame, axis=-1, name='source_frame')
            print(self.source_frame)

            self.target_frame = tf.expand_dims(self.sfm_base.target_frame, axis=-1, name='target_frame')
            print(self.target_frame)

            self.actions = self.sfm_base.actions
            print('actions: {}'.format(self.actions))

            self._init_conv_net()

            if self.sfm_base.embedding != None:
                self.embedding = self.sfm_base.embedding

            self._init_upsample_net_2()
            self._init_object_masks_and_translation()


            if self.trainable:
                self._init_transform_loss()
                self._init_vf_loss()

                self._init_action_prediction_loss()
                self._init_train_op()


    def _init_inputs(self):
        self._init_frames_placeholder(num_frames=self.hparams['num_input_frames'])


    def _init_conv_net(self):
        # Build the graph to process each input.
        activ = tf.nn.relu

        conv1_name = 'c1_2_frame_input' if self.frames.get_shape()[-1] == 2 else 'c1'

        self.conv1 = activ(conv(self.frames, conv1_name, nf=32, rf=8, stride=4, init_scale=np.sqrt(2), reuse=self.reuse, trainable=self.trainable))
        print('conv1: {}'.format(self.conv1))

        self.conv2 = activ(conv(self.conv1, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), reuse=self.reuse, trainable=self.trainable))
        print('conv2: {}'.format(self.conv2))

        self.conv3 = activ(conv(self.conv2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), reuse=self.reuse, trainable=self.trainable))
        print('conv3: {}'.format(self.conv3))

        h3 = self.conv3

        if self.hparams.get('object_segmentation_net_attention'):
            h3 = tf.nn.l2_normalize(h3, dim=-1)
            assert h3.get_shape() == (16, 7, 7, 64)

            attention_logits, _ = conv_without_bias(h3, 'attention_logits', nf=1, rf=1, stride=1, trainable=self.trainable)
            self.attention_logits = attention_logits

            orig_attention_shape = attention_logits.get_shape()
            attention_logits = tf.reshape(attention_logits, (self.hparams['batch_size'], -1))
            attention_weights = tf.nn.softmax(attention_logits)
            attention_weights = tf.reshape(attention_weights, orig_attention_shape)
            self.attention_weights = attention_weights
            print('attention_weights: {}'.format(self.attention_weights))

            self.log_image('predicted_attention', self.attention_weights)

            h3 = h3 * attention_weights * 49.0

            h3 = conv_to_fc(h3)
            print('h3: {}'.format(h3))

            self.embedding = constrained_fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2), reuse=self.reuse, trainable=self.trainable)

        else:
            h3 = conv_to_fc(h3)
            self.embedding = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2), reuse=self.reuse, trainable=self.trainable)

        self.embedding = activ(self.embedding)
        print("embedding: {}".format(self.embedding))

    def _upsample(self, value, name, output_shape, nf, rf, stride, pad='VALID'):
        if self.hparams.get('learn_upsampling'):
            result = tf.contrib.layers.conv2d_transpose(
                    inputs=value,
                    num_outputs=nf,
                    kernel_size=rf,
                    stride=stride,
                    padding=pad,
                    weights_initializer=ortho_init(np.sqrt(2)),
                    scope=name,
                    reuse=self.reuse,
                    trainable=self.trainable,
                )
        else:
            with tf.variable_scope(name):
                result = tf.image.resize_images(value, size=output_shape)

        return result


    def _init_upsample_net_2(self):
        assert not self.hparams.get('learn_upsampling')

        # if self.hparams.get('no_nonlinear'):
        #     init_scale = 1.0
        #     activ = tf.identity
        # else:
        #     init_scale = np.sqrt(2)
        #     activ = tf.nn.relu

        # 19 - depth 24, no extra concat branch.
        # 20 - depth 32, with extra concat branch.
        # 21 - both.

        depth = 24

        self.upsample_embedding = fc(self.embedding, 'fc2', nh=21*21*depth, init_scale=np.sqrt(2), reuse=self.reuse, trainable=self.trainable)
        x = tf.reshape(self.upsample_embedding, [-1,21,21,depth])
        #x = tf.nn.relu(x)


        # source_frame_downsample_4x = tf.image.resize_images(self.source_frame, size=[21, 21])
        # x = tf.concat([x, source_frame_downsample_4x], axis=-1)
        # x = conv(x, 'conv4', nf=32, rf=3, stride=1, init_scale=np.sqrt(2), pad='SAME', trainable=self.trainable, reuse=self.reuse)
        # x = tf.nn.relu(x)


        x = tf.image.resize_images(x, size=[42, 42])
        #x_img = tf.image.resize_images(x, size=[84, 84])

        #source_frame_downsample_2x = tf.image.resize_images(self.source_frame, size=[42, 42])
        #x = tf.concat([x, source_frame_downsample_2x], axis=-1)
        x = x + conv(x, 'conv5', nf=depth, rf=3, stride=1, init_scale=np.sqrt(2), pad='SAME', trainable=self.trainable, reuse=self.reuse)
        x = tf.nn.relu(x)


        x = tf.image.resize_images(x, size=[84, 84])
        #x = tf.concat([x, self.source_frame, x_img], axis=-1)
        x = x + conv(x, 'conv6', nf=depth, rf=3, stride=1, init_scale=np.sqrt(2), pad='SAME', trainable=self.trainable, reuse=self.reuse)
        x = tf.nn.relu(x)

        self.masks = conv(x, 'masks', nf=self.hparams['k_obj'], rf=1, stride=1, init_scale=1.0, trainable=self.trainable, reuse=self.reuse)

        #with tf.variable_scope('upsample1'):
        #    self.conv4 = activ(conv(x, 'conv4', nf=32, rf=3, stride=1, init_scale=init_scale, pad='SAME', trainable=self.trainable, reuse=self.reuse))
        #    print('conv4: {}'.format(self.conv4))

        # with tf.variable_scope('upsample2'):
        #     x = self.conv4
        #     x = self._upsample(x, name='u2', output_shape=[16,16], nf=64, rf=3, stride=1)
        #     self.conv5 = activ(conv(x, 'conv5', nf=32, rf=3, stride=1, init_scale=init_scale, pad='SAME', trainable=self.trainable, reuse=self.reuse))
        #     print('conv5: {}'.format(self.conv5))

        # with tf.variable_scope('upsample3'):
        #     x = self.conv4
        #     x = self._upsample(x, name='u3', output_shape=[42,42], nf=64, rf=3, stride=2)
        #     self.conv6 = x + activ(conv(x, 'conv6', nf=32, rf=3, stride=1, init_scale=init_scale, pad='SAME', trainable=self.trainable, reuse=self.reuse))
        #     print('conv6: {}'.format(self.conv6))

        # with tf.variable_scope('upsample3b'):
        #     x = self.conv6
        #     x = self._upsample(x, name='u3b', output_shape=[64,64], nf=64, rf=3, stride=2)
        #     self.conv6b = activ(conv(x, 'conv6b', nf=32, rf=3, stride=1, init_scale=init_scale, pad='SAME', trainable=self.trainable, reuse=self.reuse))
        #     print('conv6b: {}'.format(self.conv6b))

        # with tf.variable_scope('upsample4'):
        #     x = self.conv6
        #     x = self._upsample(x, name='u4', output_shape=[84,84], nf=32, rf=8, stride=4)
        #     self.conv7 = x# + activ(conv(x, 'conv7', nf=32, rf=3, stride=1, init_scale=init_scale, pad='SAME', trainable=self.trainable, reuse=self.reuse))
        #     print('conv7: {}'.format(self.conv7))


    def _init_upsample_net(self):
        activ = tf.nn.relu

        if self.hparams['fc2_upsample']:
            self.upsample_embedding = activ(fc(self.embedding, 'fc2', nh=7*7*32, init_scale=np.sqrt(2), reuse=self.reuse, trainable=self.trainable))
            print("upsample embedding: {}".format(self.upsample_embedding.get_shape()))
            x = tf.reshape(self.upsample_embedding, [-1, 7, 7, 32])

            with tf.variable_scope('upsample1'):
                if self.hparams['use_skip_connections']:
                    x = tf.concat([x, self.conv3], axis=-1)

                self.conv4 = activ(conv(x, 'conv4', nf=32, rf=3, stride=1, init_scale=np.sqrt(2), pad='SAME', trainable=self.trainable, reuse=self.reuse))
                print('conv4: {}'.format(self.conv4))
        else:
            # Add the h, w dims back in after flattening.
            self.upsample_embedding = tf.expand_dims(tf.expand_dims(self.embedding, axis=1), axis=1)
            print("upsample embedding: {}".format(self.upsample_embedding.get_shape()))

            with tf.variable_scope('upsample1'):
                x = self.upsample_embedding
                x = self._upsample(x, name='u1', output_shape=tf.shape(self.conv3)[1:3], nf=512, rf=3, stride=7)
                #x = tf.image.resize_images(x, size=tf.shape(self.conv2)[1:3])

                if self.hparams['use_skip_connections']:
                    x = tf.concat([x, self.conv3], axis=-1)

                self.conv4 = activ(conv(x, 'conv4', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), pad='SAME', trainable=self.trainable, reuse=self.reuse))
                print('conv4: {}'.format(self.conv4))

        with tf.variable_scope('upsample2'):
            x = self.conv4
            #x = tf.image.resize_images(x, size=tf.shape(self.conv1)[1:3])
            x = self._upsample(x, name='u2', output_shape=tf.shape(self.conv2)[1:3], nf=64, rf=3, stride=1)

            if self.hparams['use_skip_connections']:
                x = tf.concat([x, self.conv2], axis=-1)

            self.conv5 = activ(conv(x, 'conv5', nf=64, rf=4, stride=1, init_scale=np.sqrt(2), pad='SAME', trainable=self.trainable, reuse=self.reuse))
            print('conv5: {}'.format(self.conv5))

        with tf.variable_scope('upsample3'):
            x = self.conv5
            #x = tf.image.resize_images(x, size=tf.shape(self.conv1)[1:3])
            x = self._upsample(x, name='u3', output_shape=tf.shape(self.conv1)[1:3], nf=64, rf=4, stride=2)

            if self.hparams['use_skip_connections']:
                x = tf.concat([x, self.conv1], axis=-1)

            self.conv6 = activ(conv(x, 'conv6', nf=32, rf=4, stride=1, init_scale=np.sqrt(2), pad='SAME', trainable=self.trainable, reuse=self.reuse))
            print('conv6: {}'.format(self.conv6))

        with tf.variable_scope('upsample4'):
            x = self.conv6
            x = self._upsample(x, name='u4', output_shape=tf.shape(self.frames)[1:3], nf=32, rf=8, stride=4)

            if self.hparams['use_skip_connections'] and not self.hparams.get('single_image_glance'):
                x = tf.concat([x, self.frames], axis=-1)

            self.conv7 = activ(conv(x, 'conv7', nf=16, rf=8, stride=1, init_scale=np.sqrt(2), pad='SAME', trainable=self.trainable, reuse=self.reuse))
            print('conv7: {}'.format(self.conv7))



    def _init_object_masks_and_translation(self):
        # Create the segmentation masks.
        with tf.variable_scope('object_masks'):
            self.object_masks = tf.nn.sigmoid(self.masks, name='object_masks')
            print(self.object_masks)

            # Visualize the segmentation masks.
            for obj_index in range(self.hparams['k_obj']):
                # Convert the image to uint8 and in the range [0, 255] so we can compare each mask on a common scale.
                # Otherwise, tensorflow will automatically scale the mask for us using the highest and lowest values in
                # the mask.
                img = tf.expand_dims(self.object_masks[..., obj_index], -1)
                img = tf.cast(img * 255., 'uint8')
                self.log_image('mask_{}'.format(obj_index), img)


        # Predict the object translation.
        with tf.variable_scope('object_translation'):
            if self.hparams['do_frame_prediction']:
                # we predict one translation for each of the possible actions we could take
                x = fc(self.embedding, 'obj_t', nh=self.hparams['num_actions'] * self.hparams['k_obj'] * 2, init_scale=1.0, trainable=self.trainable, reuse=self.reuse)
                possible_object_translations = tf.reshape(x, (-1, self.hparams['num_actions'], self.hparams['k_obj'], 2), name='possible_object_translations')
                print(possible_object_translations)

                # we choose the translation based upon the ground truth action
                # old translation shape: batch x num_actions x k_obj x 2
                # new translation shape: batch x k_obj x 2 x 1 x 1

                action_mask = tf.expand_dims(tf.expand_dims(self.actions, -1), -1, name='action_mask')
                # action_mask = tf.Print(action_mask, [action_mask], message="action mask", summarize=10)
                # action_mask = tf.Print(action_mask, [self.object_translation], message="initial obj trans", summarize=50)
                print(action_mask)

                chosen_translation = tf.reduce_sum(action_mask * possible_object_translations, axis=1, name='chosen_translation')
                # chosen_translation = tf.Print(chosen_translation, [chosen_translation], message="chosen translation after sum", summarize=10)
                print(chosen_translation)

                self.object_translation = tf.identity(chosen_translation, name='object_translation')
            else:
                x = fc(self.embedding, 'obj_t', nh=self.hparams['k_obj'] * 2, init_scale=1.0, trainable=self.trainable, reuse=self.reuse)
                self.object_translation = tf.reshape(x, (-1, self.hparams['k_obj'], 2), name='object_translation')

            print(self.object_translation)

            # Background.
            #background_mask = np.ones([self.hparams['k_obj']], dtype=np.float32)
            #background_mask[0] = 0
            #self.object_translation *= tf.expand_dims(tf.expand_dims(background_mask, axis=0), axis=-1)

            self.log_scalar('abs mean', tf.reduce_mean(tf.abs(self.object_translation)))
            self.log_scalar('min', tf.reduce_min(self.object_translation))
            self.log_scalar('max', tf.reduce_max(self.object_translation))


    def _init_transform_loss(self):
        # Create a 'default' sampling grid.
        with tf.variable_scope('mesh_grid'):
            x_linspace = tf.linspace(0., self.hparams['img_w'] - 1., self.hparams['img_w'])
            y_linspace = tf.linspace(0., self.hparams['img_h'] - 1., self.hparams['img_h'])

            x_coord, y_coord = tf.meshgrid(x_linspace, y_linspace)
            print("x_coord: {}".format(x_coord))
            print("y_coord: {}".format(y_coord))

            x_coord = tf.reshape(x_coord, [1, -1], name='flat_x_coords')
            y_coord = tf.reshape(y_coord, [1, -1], name='flat_y_coords')
            print(y_coord)
            print(x_coord)

            self.mesh_grid = tf.concat([y_coord, x_coord], axis=0, name='mesh_grid')
            print(self.mesh_grid)


        # Calculate where to lookup pixels.
        with tf.variable_scope('flow'):
            # new_shape: batch x k_obj x 1 x h x w
            obj_mask = tf.transpose(self.object_masks, perm=(0, 3, 1, 2))
            obj_mask = tf.expand_dims(obj_mask, 2, name='expanded_obj_mask')
            print(obj_mask)

            obj_t = tf.expand_dims(tf.expand_dims(self.object_translation, -1), -1, name='expanded_obj_t')
            print(obj_t)

            # batch x k_obj x 2 x h x w
            translation_masks = tf.multiply(obj_mask, obj_t, name='translation_masks')
            print(translation_masks)

            self.translation_masks = translation_masks

            self.flow = tf.reduce_sum(translation_masks, axis=1, name='flow')
            print(self.flow)

            flattened_flow = tf.reshape(self.flow, [-1, 2, self.hparams['img_w'] * self.hparams['img_h']], name='flattened_flow')
            print(flattened_flow)

            if self.hparams['model_camera_motion']:
                self.camera_translation = fc(self.embedding, 'camera_translation', nh=2, init_scale=0.0, trainable=self.trainable, reuse=False)
                print(self.camera_translation)
                flattened_flow += tf.expand_dims(self.camera_translation, axis=-1)

                self.log_scalar('camera abs mean', tf.reduce_mean(tf.abs(self.camera_translation)))
                self.log_scalar('camera min', tf.reduce_min(self.camera_translation))
                self.log_scalar('camera max', tf.reduce_max(self.camera_translation))

            # Add in the default coordinates.
            # TODO(vikgoel): could we instead just change the init scale of obj_t?
            img_size_f = self.hparams['flow_c'] * np.array([[self.hparams['img_h']], [self.hparams['img_w']]], dtype=np.float32)
            print("img_size_f: {}".format(img_size_f))

            self.sampling_coords = tf.add(img_size_f * flattened_flow, self.mesh_grid, name='sampling_coords')
            print(self.sampling_coords)

            # TODO(vikgoel): better visualization of flow.
            self.log_image('flow_y', tf.expand_dims(self.flow[:, 0, :, :], -1))
            self.log_image('flow_x', tf.expand_dims(self.flow[:, 1, :, :], -1))


        with tf.variable_scope('frame_loss'):
            y_s = self.sampling_coords[:, 0, :]
            x_s = self.sampling_coords[:, 1, :]

            print("y_s: {}".format(y_s))
            print("x_s: {}".format(x_s))

            y_s_flatten = tf.reshape(y_s, [-1])
            x_s_flatten = tf.reshape(x_s, [-1])

            print("y_s_flatten: {}".format(y_s_flatten))
            print("x_s_flatten: {}".format(x_s_flatten))

            self.transformed_image = self._interpolate(self.source_frame, x_s_flatten, y_s_flatten, self.batch_dims[1:3])
            self.transformed_image = tf.reshape(self.transformed_image, self.batch_dims)

            self.log_image('transformed_image', self.transformed_image)
            self.log_image('loss image', self.transformed_image - self.target_frame)

            self.transform_loss = self._compute_transform_loss(self.target_frame, self.transformed_image)
            self.no_transform_loss = self._compute_transform_loss(self.target_frame, self.source_frame)
            self.relative_error = self.transform_loss / (self.no_transform_loss + 1e-7)

            self.log_scalar('transform_loss', tf.reduce_mean(self.transform_loss))
            self.log_scalar('no_transform_loss', tf.reduce_mean(self.no_transform_loss))
            self.log_scalar('relative_error', tf.reduce_mean(self.relative_error))

            def log_mask(name, mask):
                print('{}: {}'.format(name, mask))

                with tf.variable_scope(name):
                    stats = tf.reduce_sum(tf.cast(mask, 'float32'), axis=[1, 2, 3])
                    self.log_scalar('min', tf.reduce_min(stats))
                    self.log_scalar('max', tf.reduce_max(stats))
                    self.log_scalar('mean', tf.reduce_mean(stats))

            incorrect_pixel_mask = tf.abs(tf.subtract(self.source_frame, self.transformed_image)) > 1e-6
            different_pixel_mask = tf.logical_not(self.same_pixel_mask)
            learnable_pixel_mask = tf.logical_and(incorrect_pixel_mask, different_pixel_mask)

            log_mask('incorrect_pixel_mask', incorrect_pixel_mask)
            log_mask('different_pixel_mask', different_pixel_mask)
            log_mask('learnable_pixel_mask', learnable_pixel_mask)


    def _init_vf_loss(self):
        if not self.hparams.get('pretrain_vf'): return

        self.vf = fc(self.embedding, 'v', 1, init_scale=1.0)[:,0]
        print('vf: {}'.format(self.vf.get_shape()))

        vf_mean, vf_var = tf.nn.moments(self.vf, axes=[0])

        self.log_scalar('vf_mean', vf_mean)
        self.log_scalar('vf_var', vf_var)

        self.value_placeholder = tf.placeholder(shape=[self.hparams['batch_size']], dtype=tf.float32, name='value_placeholder')

        self.vf_loss = tf.reduce_mean(mse(self.vf, self.value_placeholder))

        if not self.is_teacher_network:
            self.log_scalar('vf_loss', self.vf_loss)
            self.log_scalar('zero_vf_loss', 0.5 * tf.reduce_mean(self.value_placeholder**2))


    def _init_action_prediction_loss(self):
        pass
        # self.action_logits = fc(self.embedding, 'action_prediction', nh=3, init_scale=np.sqrt(2))
        # self.action_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.action_logits, labels=self.ground_truth_actions)
        # self.action_loss = tf.reduce_mean(self.action_loss)
        # self.action_loss = self.action_loss * self.hparams['action_loss_c']
        # self.log_scalar('action_loss', self.action_loss)

        # accuracy = tf.equal(tf.argmax(self.action_logits, axis=-1), tf.argmax(self.ground_truth_actions, axis=-1))
        # self.log_scalar('action_prediction_accuracy', tf.reduce_mean(tf.cast(accuracy, tf.float32)))


    def _init_train_op(self):
        # We want the mask to cluster around an object, and be zero everyone else.
        # By using L1 regularization perhaps we can encourage this desired sparsity.
        with tf.variable_scope('object_mask_regularization'):
            # TODO(vikgoel): we currently use a manhattan distance here, maybe euclidian would be better?
            self.mask_reg_c = tf.placeholder(dtype=tf.float32, shape=(), name='mask_reg_c')

            assert len(self.translation_masks.get_shape()) == 5
            unscaled_mask_reg_loss = tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(tf.abs(self.translation_masks), axis=-1), axis=-1), axis=-1), axis=-1)

            self.mask_reg_loss = self.mask_reg_c * unscaled_mask_reg_loss
            self.log_scalar('mask_reg_c', self.mask_reg_c)
            self.log_scalar('mask_reg_loss', tf.reduce_mean(unscaled_mask_reg_loss))

        with tf.variable_scope('embedding_reg'):
            self.embedding_reg_loss = self.hparams['embedding_reg_loss_c'] * tf.reduce_mean(tf.abs(self.embedding))
            self.log_scalar('embedding_reg_loss', self.embedding_reg_loss)

        # L2 weight decay
        self.loss_l2 = tf.multiply(
            tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name ]),
            self.hparams['l2_reg_c'],
            name='loss_l2',
        )
        print(self.loss_l2)

        # Entropy loss
        self.entropy_loss = 0.0
        if hasattr(self, 'attention_logits'):
            flattened_logits = tf.reshape(self.attention_logits, [self.hparams['batch_size'], -1])
            assert flattened_logits.get_shape()[-1] == 49
            attention_entropy = tf.reduce_mean(cat_entropy(flattened_logits))
            self.entropy_loss = self.hparams['attention_entropy_c'] * attention_entropy
            self.log_scalar('attention_entropy', attention_entropy)

        with tf.variable_scope('train'):
            unscaled_total_loss = self.transform_loss + self.mask_reg_loss + self.loss_l2 + self.entropy_loss + self.embedding_reg_loss

            if hasattr(self, 'vf_loss'):
                unscaled_total_loss += self.vf_loss

            self.total_loss = unscaled_total_loss * self.hparams['lr_mult']

        self.train_op = self._create_train_op(loss=tf.reduce_mean(self.total_loss))


    def _compute_transform_loss(self, truth_image, predicted_image):
        #loss = tf.subtract(truth_image, predicted_image)
        #loss = tf.reduce_mean(tf.abs(loss))
        #return loss
        loss = (1.0 - tf.image.ssim(truth_image, predicted_image, max_val=1.0)) / 2.0
        return loss

    # taken from https://github.com/daviddao/spatial-transformer-tensorflow/blob/master/spatial_transformer.py
    def _repeat(self, x, n_repeats):
        with tf.variable_scope('_repeat'):
            rep = tf.transpose(
                tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
            rep = tf.cast(rep, 'int32')
            x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
            return tf.reshape(x, [-1])

    # taken from https://github.com/daviddao/spatial-transformer-tensorflow/blob/master/spatial_transformer.py
    def _interpolate(self, im, x, y, out_size):
        with tf.variable_scope('_interpolate'):
            # constants
            num_batch = tf.shape(im)[0]
            height = tf.shape(im)[1]
            width = tf.shape(im)[2]
            channels = tf.shape(im)[3]

            x = tf.cast(x, 'float32')
            y = tf.cast(y, 'float32')
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]
            zero = tf.zeros([], dtype='int32')
            max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
            max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')

            # do sampling
            x0 = tf.cast(tf.floor(x), 'int32')
            x0 = tf.clip_by_value(x0, zero, max_x)
            unclipped_x1 = x0 + 1
            x1 = tf.clip_by_value(unclipped_x1, zero, max_x)

            y0 = tf.cast(tf.floor(y), 'int32')
            y0 = tf.clip_by_value(y0, zero, max_y)
            unclipped_y1 = y0 + 1
            y1 = tf.clip_by_value(unclipped_y1, zero, max_y)

            dim2 = width
            dim1 = width*height
            base = self._repeat(tf.range(num_batch)*dim1, out_height*out_width)
            base_y0 = base + y0*dim2
            base_y1 = base + y1*dim2
            idx_a = base_y0 + x0
            idx_b = base_y1 + x0
            idx_c = base_y0 + x1
            idx_d = base_y1 + x1

            # use indices to lookup pixels in the flat image and restore
            # channels dim
            im_flat = tf.reshape(im, tf.stack([-1, channels]))
            im_flat = tf.cast(im_flat, 'float32')
            Ia = tf.gather(im_flat, idx_a)
            Ib = tf.gather(im_flat, idx_b)
            Ic = tf.gather(im_flat, idx_c)
            Id = tf.gather(im_flat, idx_d)

            unclipped_x1 = tf.clip_by_value(unclipped_x1, zero, max_x + 1)
            unclipped_y1 = tf.clip_by_value(unclipped_y1, zero, max_y + 1)

            x = tf.clip_by_value(x, 0., tf.cast(max_x, tf.float32))
            y = tf.clip_by_value(y, 0., tf.cast(max_y, tf.float32))

            # and finally calculate interpolated values
            x0_f = tf.cast(x0, 'float32')
            x1_f = tf.cast(unclipped_x1, 'float32')
            y0_f = tf.cast(y0, 'float32')
            y1_f = tf.cast(unclipped_y1, 'float32')
            wa = tf.expand_dims(((x1_f-x) * (y1_f-y)), 1)
            wb = tf.expand_dims(((x1_f-x) * (y-y0_f)), 1)
            wc = tf.expand_dims(((x-x0_f) * (y1_f-y)), 1)
            wd = tf.expand_dims(((x-x0_f) * (y-y0_f)), 1)
            output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])

            Ia = tf.reshape(Ia, self.batch_dims)
            Ib = tf.reshape(Ib, self.batch_dims)
            Ic = tf.reshape(Ic, self.batch_dims)
            Id = tf.reshape(Id, self.batch_dims)

            self.same_pixel_mask = tf.logical_and(
                                        tf.logical_and(
                                            tf.equal(Ia, Ib), tf.equal(Ib, Ic)),
                                        tf.equal(Ic, Id))
            return output


class ObjectSegmentationNetworkPredict(ObjectSegmentationNetworkCommon):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # for now we don't support sfm_base
        assert(self.sfm_base is None)
        assert(self.hparams['num_prediction_frames'] == 1)

        self._init_inputs()

        sfm_base = ObjectSegmentationBase(
            frames=self.frames[..., :self.hparams['num_input_frames']],
            target_frame=self.frames[..., self.hparams['num_input_frames']],
            source_frame=self.frames[..., self.hparams['num_input_frames'] - 1],
            actions=self.actions[:, self.hparams['num_input_frames'] - 1, :],
        )
        kwargs['sfm_base'] = sfm_base
        self.forwards_net = ObjectSegmentationNetwork(**kwargs)

        sfm_base.target_frame = self.frames[..., self.hparams['num_input_frames'] - 2]
        kwargs['sfm_base'] = sfm_base
        kwargs['reuse'] = True
        self.backwards_net = ObjectSegmentationNetwork(**kwargs)

        self.total_loss = self.forwards_net.total_loss + self.backwards_net.total_loss
        self.train_op = self._create_train_op(self.total_loss)


    def _init_inputs(self):
        self.num_timesteps = self.hparams['num_input_frames'] + self.hparams['num_prediction_frames']
        self._init_frames_placeholder(num_frames=self.num_timesteps)

        with tf.variable_scope('placeholders'):
            self.actions = tf.placeholder(
                tf.float32,
                (self.hparams['batch_size'], self.num_timesteps, self.hparams['num_actions']),
                name='actions',
            )
            print(self.actions)

