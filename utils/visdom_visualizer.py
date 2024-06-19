from visdom import Visdom
import numpy as np

#  szm    | RUNNING | 10.57.253.203 (eth0)
# iptables -t nat -A PREROUTING -p tcp --dport 11111 -j DNAT --to 10.57.253.203:8097
# iptables -A INPUT -p tcp --dport 11111 -j ACCEPT
class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main', **kwargs):
        self.viz = Visdom(env=env_name, use_incoming_socket=True, **kwargs)
        self.env = env_name
        self.plots = {}
        self.index = {}
    def plot(self, var_name, split_name, lr, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title='lr='+str(lr),
                xlabel='steps',
                ylabel=var_name, 
                width=300,
                height=240
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update='append', opts=dict(
                title='lr='+str(lr)
            ))

    def plot_epoch(self, var_name, split_name, title_name, x, y):
        """_summary_  https://github.com/noagarcia/visdom-tutorial
            var_name: variable name (e.g. loss, acc)
            split_name: split name (e.g. train, val)
            title_name: titles of the graph (e.g. Classification Accuracy)
            x: x axis value (e.g. epoch number)
            y: y axis value (e.g. epoch loss)
            
            example:
            In the training function we add the loss value after every epoch as:
            plotter.plot('loss', 'train', 'Class Loss', epoch, losses.avg)

            In the validation function we add the loss and the accuracy values as:
            plotter.plot('loss', 'val', 'Class Loss', epoch, losses.avg)
            plotter.plot('acc', 'val', 'Class Accuracy', epoch, acc)
        """
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name,
                width=350,
                height=240
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')

    def merge(self, image_name, epoch, image_tensor):
        if not isinstance(image_tensor, np.ndarray):
            image = image_tensor.detach().cpu().numpy()
        with np.errstate(invalid='ignore'):
            image = 255 * (image - np.min(image)) / (np.max(image) - np.min(image))
        if isinstance(epoch, int):
            title = image_name + '_Epoch' +str(epoch)
        else:
            title = image_name + '_loss' +str(epoch)
        if image_name not in self.plots:
            self.plots[image_name] = self.viz.image(image, env=self.env, opts=dict(
                title = title
            ))
        else:
            self.viz.image(image, env=self.env, win=self.plots[image_name], opts=dict(title=title))

    def image(self, image_name, epoch, image_tensor):
        if not isinstance(image_tensor, np.ndarray):
            image = image_tensor.detach().cpu().numpy()
        else:
            image = image_tensor
        with np.errstate(invalid='ignore'):
            image = 255 * (image - np.min(image)) / (np.max(image) - np.min(image))
        if isinstance(epoch, int):
            title = image_name + '_Epoch' +str(epoch)
        else:
            title = image_name + '_loss' +str(epoch)
        if image_name not in self.plots:
            self.plots[image_name] = self.viz.image(image[0, 0, ...], env=self.env, opts=dict(
                title = title
            ))
        else:
            self.viz.image(image[0, 0, ...], env=self.env, win=self.plots[image_name], opts=dict(title=title))

    def mask(self, image_name, epoch, image_tensor):
        if not isinstance(image_tensor, np.ndarray):
            image = image_tensor.detach().cpu().numpy()[0:1,...]
        if image.shape[2] == 1:
            image = np.repeat(image,image.shape[-1], 2)
        if image.shape[3] == 1:
            image = np.repeat(image,image.shape[-2], 3)
        image = np.fft.fftshift(image)
        # image = np.fft.fftshift(image_tensor)
        image = 255 * (image - np.min(image)) / (np.max(image) - np.min(image))
        # image = np.fft.fftshift(image)  # add
        # outfile = "/home/labuser1/wzw/COMBINE_PYTORCH/models/trajectory_motion/traj_{}".format(epoch)
        # np.save(outfile, image)        
        title = image_name + '_Epoch' +str(epoch)
        if image_name not in self.plots:
            self.plots[image_name] = self.viz.image(image[0, 0, ...], env=self.env, opts=dict(
                title = title
            ))
        else:
            self.viz.image(image[0, 0, ...], env=self.env, win=self.plots[image_name], opts=dict(title=title))

    def one_piece_mask(self, image_name, epoch, image_tensor):
        if np.size(image_tensor.shape) == 5:
            image_tensor = image_tensor[0,0,:,:,0]
        if np.size(image_tensor.shape) == 4:
            image_tensor = image_tensor[0,0,:,:]
        if np.size(image_tensor.shape) == 3:
            image = image_tensor[0,:,:]
        if not isinstance(image_tensor, np.ndarray):
            image = image_tensor.detach().cpu().numpy()

        image = np.fft.fftshift(image)
        # image = np.fft.fftshift(image_tensor)
        image = 255 * (image - np.min(image)) / (np.max(image) - np.min(image))
        

        # image = np.fft.fftshift(image)  # add
        # outfile = "/home/labuser1/wzw/COMBINE_PYTORCH/models/trajectory_motion/traj_{}".format(epoch)
        # np.save(outfile, image)        
        title = image_name + '_Epoch' +str(epoch)
        if image_name not in self.plots:
            self.plots[image_name] = self.viz.image(image, env=self.env, opts=dict(
                title = title
            ))
        else:
            self.viz.image(image, env=self.env, win=self.plots[image_name], opts=dict(title=title))

    def k_space(self, image_name, epoch, image_tensor):
        image = image_tensor.detach().cpu().numpy()
        image_complex = 1j * image[0, 1, ...]
        image_complex += image[0, 0, ...]
        # image = np.log(np.fft.fftshift(np.abs(image_complex)))
        image = np.fft.fftshift(np.log(np.abs(image_complex)))
        image[image == np.inf] = 0
        image[image == -np.inf] = 0
        image = 255 * (image - np.min(image)) / (np.max(image) - np.min(image))

        # image = np.fft.fftshift(image)  # add

        title = image_name + '_Epoch' +str(epoch)
        if image_name not in self.plots:
            self.plots[image_name] = self.viz.image(image, env=self.env, opts=dict(
                title = title
            ))
        else:
            self.viz.image(image, env=self.env, win=self.plots[image_name], opts=dict(title=title))

    def seg_image(self, image_name, epoch, image_tensor):
        image = image_tensor.detach().cpu().numpy()
        B, C, H, W = image_tensor.shape
        seg_map = np.zeros_like(image)[0, 0, ...]
        for c in range(C):
            if c == 0:
                continue # c == 0 时，1是背景，0是其他。
            temp = image[0, c, ...]
            temp = temp > 0.5
            temp = temp *(c+1)*20
            seg_map += temp
        # seg_map = seg_map *30
        if isinstance(epoch, int):
            title = image_name + '_Epoch' +str(epoch)
        else:
            title = image_name + '_loss' +str(epoch)
        if image_name not in self.plots:
            self.plots[image_name] = self.viz.image(seg_map, env=self.env, opts=dict(
                title = title
            ))
        else:
            self.viz.image(seg_map, env=self.env, win=self.plots[image_name], opts=dict(title=title))

    def seg_image_split(self, image_name, epoch, image_tensor):
        image = image_tensor.detach().cpu().numpy()
        B, C, H, W = image_tensor.shape
        # seg_map = np.zeros_like(image)[0, 0, ...]
        seg_map = np.empty_like(image)[0, 0, ...]
        for c in range(0, C):
            temp = image[0, c, ...]
            temp[temp >= 0.5] = 1
            temp[temp < 0.5] = 0
            temp = temp *200
            if c==0:
                seg_map = temp
            else:    
                seg_map = np.concatenate((seg_map, temp), 1)
        # seg_map = seg_map *30
        if isinstance(epoch, int):
            title = image_name + '_Epoch' +str(epoch)
        else:
            title = image_name + '_loss' +str(epoch)
        if image_name not in self.plots:
            self.plots[image_name] = self.viz.image(seg_map, env=self.env, opts=dict(
                title = title
            ))
        else:
            self.viz.image(seg_map, env=self.env, win=self.plots[image_name], opts=dict(title=title))

    def simple_img(self, patient_slice, image, image_name='test'):
        title = str(patient_slice)
        if image_name not in self.plots:
            self.plots[image_name] = self.viz.image(image, env=self.env, opts=dict(
                title = title,
                width=500,
                height=500
            ))
        else:
            self.viz.image(image, env=self.env, win=self.plots[image_name], opts=dict(title=title))

    def merge_recon_and_seg(self, image, seg):
        pass

    def plot_loss(self, name, y, **kwargs):
        """
        self.plot('loss',1.00)
        """
        x = self.index.get(name, 0)
        self.viz.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1