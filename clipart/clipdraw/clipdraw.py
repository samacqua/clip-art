from .helpers import *
from .nouns import noun_prompts, nouns

import clip
import torch
from torchvision import transforms
import pydiffvg
import random

from PIL import Image

def clip_draw_render(clip_model, prompt, width=600, height=600, num_paths=256,
                     num_rects=0, num_circs=0, num_iter=100, max_width=50,
                     exp_dir='clipdraw_out', use_normalized_clip=True, init_image=None,
                     target_img=None):

    assert torch.cuda.is_available(), 'need GPU'

    os.makedirs(exp_dir, exist_ok=True)
    device = torch.device('cuda:0')
    nouns_features = clip_model.encode_text(torch.cat([clip.tokenize(noun_prompts).to(device)]))

    # Calculate features
    with torch.no_grad():
        text_input = clip.tokenize(prompt).to(device)
        text_features = clip_model.encode_text(text_input)

    pydiffvg.set_print_timing(False)
    gamma = 1.0

    # Use GPU if available
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    pydiffvg.set_device(device)

    canvas_width, canvas_height = 224, 224

    if target_img is not None:
        target_img = Image.open(target_img)
        target_img = target_img.resize((canvas_width, canvas_height))
        target_img = torch.tensor(np.array(target_img)) / 255
        target_img = target_img.permute(2, 0, 1).to(device)

    # Image Augmentation Transformation
    augment_trans = transforms.Compose([
        transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.5),
        transforms.RandomResizedCrop(224, scale=(0.7, 0.9)),
    ])

    if use_normalized_clip:
        augment_trans = transforms.Compose([
            transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.5),
            transforms.RandomResizedCrop(224, scale=(0.7, 0.9)),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

    # Initialize Random Curves
    curves = []
    curve_groups = []
    for i in range(num_paths):
        num_segments = random.randint(1, 3)
        num_control_points = torch.zeros(num_segments, dtype=torch.int32) + 2
        points = []
        p0 = (random.random(), random.random())
        points.append(p0)
        for j in range(num_segments):
            radius = 0.1
            p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
            p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
            p3 = (p2[0] + radius * (random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
            points.append(p1)
            points.append(p2)
            points.append(p3)
            p0 = p3
        points = torch.tensor(points)
        points[:, 0] *= canvas_width
        points[:, 1] *= canvas_height
        path = pydiffvg.Path(num_control_points=num_control_points, points=points, stroke_width=torch.tensor(1.0),
                             is_closed=False)
        curves.append(path)
        path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(curves) - 1]), fill_color=None,
                                         stroke_color=torch.tensor(
                                             [random.random(), random.random(), random.random(), random.random()]))
        curve_groups.append(path_group)

    # intialize random rectangles
    rects = []
    rect_groups = []
    for i in range(num_rects):
        p_min_h, p_min_w = np.random.rand() * canvas_height, np.random.rand() * canvas_width
        p_max_h = np.random.uniform(low=p_min_h+5, high=p_min_h+50)
        p_max_w = np.random.uniform(low=p_min_w+5, high=p_min_w+50)
        rect = pydiffvg.Rect(p_min=torch.tensor([p_min_h, p_min_w]), p_max=torch.tensor([p_max_h, p_max_w]), stroke_width = torch.tensor(0.))
        rects.append(rect)
        path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([num_paths + len(rects) - 1]), fill_color = torch.tensor([random.random(), random.random(), random.random(), random.random()]))
        rect_groups.append(path_group)

    # intialize random circles
    circs = []
    circ_groups = []
    for i in range(num_circs):
        radius = np.random.rand() * min(canvas_width, canvas_height)
        center_x, center_y = np.random.rand() * canvas_width, np.random.rand() * height
        circ = pydiffvg.Circle(radius=torch.tensor(radius), center=torch.tensor([center_y, center_x]), stroke_width=torch.tensor(1.0))
        circs.append(circ)
        path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([num_paths + num_rects + len(circs) - 1]), fill_color = torch.tensor([random.random(), random.random(), random.random(), random.random()]))
        circ_groups.append(path_group)

    # Just some diffvg setup
    render = pydiffvg.RenderFunction.apply

    points_vars = []
    stroke_width_vars = []
    color_vars = []
    for path in curves:
        path.points.requires_grad = True
        points_vars.append(path.points)
        path.stroke_width.requires_grad = True
        stroke_width_vars.append(path.stroke_width)
    for group in curve_groups:
        group.stroke_color.requires_grad = True
        color_vars.append(group.stroke_color)

    rect_min_vars = []
    rect_max_vars = []
    for rect in rects:
        rect.p_min.requires_grad = True
        rect_min_vars.append(rect.p_min)
        rect.p_max.requires_grad = True
        rect_max_vars.append(rect.p_max)
    for group in rect_groups:
        group.fill_color.requires_grad = True
        color_vars.append(group.fill_color)

    circ_center_vars = []
    circ_radius_vars = []
    for circ in circs:
        circ.center.requires_grad = True
        circ_center_vars.append(circ.center)
        circ.radius.requires_grad = True
        circ_radius_vars.append(circ.radius)
    for group in circ_groups:
        group.fill_color.requires_grad = True
        color_vars.append(group.fill_color)

    # Optimizers
    rects_lr = 10.0
    circs_lr = 1.0
    if num_paths > 0:
        points_optim = torch.optim.Adam(points_vars, lr=1.0)
        width_optim = torch.optim.Adam(stroke_width_vars, lr=0.1)
    if num_rects > 0:
        rect_min_optim = torch.optim.Adam(rect_min_vars, lr=rects_lr)
        rect_max_optim = torch.optim.Adam(rect_max_vars, lr=rects_lr)
    if num_circs > 0:
        circ_center_optim = torch.optim.Adam(circ_center_vars, lr=circs_lr)
        circ_radius_optim = torch.optim.Adam(circ_radius_vars, lr=circs_lr)
    color_optim = torch.optim.Adam(color_vars, lr=0.01)

    # Run the main optimization loop
    for t in range(num_iter):

        # Anneal learning rate (makes videos look cleaner)
        if t == int(num_iter * 0.5):
            if num_paths > 0:
                for g in points_optim.param_groups:
                    g['lr'] = 0.4
            if num_rects > 0:
                for g in rect_min_optim.param_groups:
                    g['lr'] = 0.4 * rects_lr
                for g in rect_max_optim.param_groups:
                    g['lr'] = 0.4 * rects_lr
            if num_circs > 0:
                for g in circ_center_optim.param_groups:
                    g['lr'] = 0.4 * circs_lr
                for g in circ_radius_optim.param_groups:
                    g['lr'] = 0.4 * circs_lr
        if t == int(num_iter * 0.75):
            if num_paths > 0:
                for g in points_optim.param_groups:
                    g['lr'] = 0.1
            if num_rects > 0:
                for g in rect_min_optim.param_groups:
                    g['lr'] = 0.1 * rects_lr
                for g in rect_max_optim.param_groups:
                    g['lr'] = 0.1 * rects_lr
            if  num_circs > 0:
                for g in circ_center_optim.param_groups:
                    g['lr'] = 0.1 * circs_lr
                for g in circ_radius_optim.param_groups:
                    g['lr'] = 0.1 * circs_lr

        if num_paths > 0:
            points_optim.zero_grad()
            width_optim.zero_grad()
        if num_rects > 0:
            rect_min_optim.zero_grad()
            rect_max_optim.zero_grad()
        if num_circs > 0:
            circ_radius_optim.zero_grad()
            circ_center_optim.zero_grad()
        color_optim.zero_grad()

        scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, canvas_height, curves+rects+circs, curve_groups+rect_groups+circ_groups)
        img = render(canvas_width, canvas_height, 2, 2, t, None, *scene_args)
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3,
                                                          device=pydiffvg.get_device()) * (1 - img[:, :, 3:4])

        # export higher quality render
        exp_img = render(width, height, 2, 2, t, None, *scene_args)
        exp_img = exp_img[:, :, 3:4] * exp_img[:, :, :3] + torch.ones(exp_img.shape[0], exp_img.shape[1], 3,
                                                                      device=pydiffvg.get_device()) * (
                              1 - exp_img[:, :, 3:4])
        save_dir = os.path.join(exp_dir, 'iter_{}.png'.format(t))
        pydiffvg.imwrite(exp_img.cpu(), save_dir, gamma=gamma)

        img = img[:, :, :3]
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2)  # NHWC -> NCHW

        loss = 0
        NUM_AUGS = 4
        img_augs = []
        for n in range(NUM_AUGS):
            img_augs.append(augment_trans(img))
        im_batch = torch.cat(img_augs)
        image_features = clip_model.encode_image(im_batch)
        for n in range(NUM_AUGS):
            loss -= torch.cosine_similarity(text_features, image_features[n:n + 1], dim=1)
        clip_loss = -loss

        target_loss = 0
        mse_loss_scale = 0.5
        if target_img is not None:
            mse_loss = torch.nn.MSELoss()
            for n in range(NUM_AUGS):
                loss += mse_loss(img_augs[n].squeeze(), target_img) * mse_loss_scale
                target_loss += mse_loss(img_augs[n].squeeze(), target_img) * mse_loss_scale

        # print('clip:', clip_loss/(clip_loss+target_loss))
        # print('target:', target_loss/(clip_loss+target_loss))

        # Backpropagate the gradients.
        loss.backward()

        # Take a gradient descent step.
        if num_paths > 0:
            points_optim.step()
            width_optim.step()
        if num_rects > 0:
            rect_min_optim.step()
            rect_max_optim.step()
        if num_circs > 0:
            circ_center_optim.step()
            circ_center_optim.step()
        color_optim.step()

        for path in curves:
            path.stroke_width.data.clamp_(1.0, max_width)
        for group in curve_groups:
            group.stroke_color.data.clamp_(0.0, 1.0)
        for rect in rects:
            rect.p_min.data.clamp_(0.0, canvas_width)
            rect.p_max.data.clamp_(rect.p_min+20, rect.p_min+50)
        for group in rect_groups:
            group.fill_color.data.clamp_(0.0, 1.0)
        for circ in circs:
            circ.radius.data.clamp_(0.0, canvas_width/2)
            circ.center.data.clamp_(0.0, max(canvas_width, canvas_height))
        for group in circ_groups:
            group.fill_color.data.clamp_(0.0, 1.0)

        if t % 10 == 0:
            show_img(img.detach().cpu().numpy()[0])
            # show_img(torch.cat([img.detach(), img_aug.detach()], axis=3).cpu().numpy()[0])
            print('render loss:', loss.item())
            print('iteration:', t)
            with torch.no_grad():
                im_norm = image_features / image_features.norm(dim=-1, keepdim=True)
                noun_norm = nouns_features / nouns_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * im_norm @ noun_norm.T).softmax(dim=-1)
                values, indices = similarity[0].topk(5)
                print("\nTop predictions:\n")
                for value, index in zip(values, indices):
                    print(f"{nouns[index]:>16s}: {100 * value.item():.2f}%")

    fnames = [os.path.join(exp_dir, f'iter_{t}.png') for t in range(num_iter)]
    images = [Image.open(f) for f in fnames]
    return images

def clip_draw_single_line(clip_model, prompt, line_complexity=12, line_width=0.6, use_normalized_clip=True, num_iter=200,
                          width=600, height=600, exp_dir='clipdraw_single_out'):
    device = torch.device('cuda:0')

    num_paths = 1
    radius = 0.025  # 0.02 - 0.1
    num_augs = 1
    prompt = f"Professional one line drawing art of a {prompt}. Professional one line sketch of {prompt}. minimalistic art tattoo"
    n_lines = (line_complexity,) * 2
    nouns_features = clip_model.encode_text(torch.cat([clip.tokenize(noun_prompts).to(device)]))

    # Calculate features
    with torch.no_grad():
        text_input = clip.tokenize(prompt).to(device)
        text_features = clip_model.encode_text(text_input)

    pydiffvg.set_print_timing(False)

    gamma = 1.0

    # Use GPU if available
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    pydiffvg.set_device(device)

    canvas_width, canvas_height = 224, 224  # 224, 224
    max_width = line_width

    # Image Augmentation Transformation
    augment_trans = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 0.9)),
    ])

    if use_normalized_clip:
        augment_trans = transforms.Compose([
            transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.5),
            transforms.RandomResizedCrop(224, scale=(0.7, 0.9)),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

    # Initialize Random Curves
    shapes = []
    shape_groups = []
    for i in range(num_paths):
        num_segments = random.randint(n_lines[0], n_lines[1])
        num_control_points = torch.zeros(num_segments, dtype=torch.int32) + 2
        points = []
        p0 = (0.5, 0.5)  # (0.5, 0.5) # (random.random(), random.random())
        points.append(p0)
        for j in range(num_segments):
            radius = radius
            p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
            p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
            p3 = (p2[0] + radius * (random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
            points.append(p1)
            points.append(p2)
            points.append(p3)
            p0 = p3
        points = torch.tensor(points)
        points[:, 0] *= canvas_width
        points[:, 1] *= canvas_height
        path = pydiffvg.Path(num_control_points=num_control_points, points=points, stroke_width=torch.tensor(0.2),
                             is_closed=False)  # stroke_width = torch.tensor(1.0) or torch.tensor(.05)
        shapes.append(path)
        path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1]), fill_color=None,
                                         stroke_color=torch.tensor([0.02, 0.02, 0.02,
                                                                    1.]))  # [random.random(), random.random(), random.random(), random.random() # [0.094, 0.310, 0.635, 1.]
        shape_groups.append(path_group)

    # Just some diffvg setup
    render = pydiffvg.RenderFunction.apply

    points_vars = []
    stroke_width_vars = []
    color_vars = []
    for path in shapes:
        path.points.requires_grad = True
        points_vars.append(path.points)
        path.stroke_width.requires_grad = True
        stroke_width_vars.append(path.stroke_width)
    for group in shape_groups:
        group.stroke_color.requires_grad = False  # True
        color_vars.append(group.stroke_color)

    # Optimizers
    points_optim = torch.optim.Adam(points_vars, lr=1.6)  # 1.0
    width_optim = torch.optim.Adam(stroke_width_vars, lr=0.1)
    color_optim = torch.optim.Adam(color_vars, lr=0.01)

    # Run the main optimization loop
    for t in range(num_iter):

        # Anneal learning rate (makes videos look cleaner)
        if t == int(num_iter * 0.15):
            for g in points_optim.param_groups:
                g['lr'] = 1.1
        if t == int(num_iter * 0.4):
            for g in points_optim.param_groups:
                g['lr'] = 0.8
        if t == int(num_iter * 0.8):
            for g in points_optim.param_groups:
                g['lr'] = 0.5
        if t == int(num_iter * 0.9):
            for g in points_optim.param_groups:
                g['lr'] = 0.3

        points_optim.zero_grad()
        width_optim.zero_grad()
        color_optim.zero_grad()
        scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, canvas_height, shapes, shape_groups)
        img = render(canvas_width, canvas_height, 2, 2, t, None, *scene_args)
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3,
                                                          device=pydiffvg.get_device()) * (1 - img[:, :, 3:4])

        # save image in higher quality render
        hd_img = render(width, height, 2, 2, t, None, *scene_args)
        hd_img = hd_img[:, :, 3:4] * hd_img[:, :, :3] + torch.ones(hd_img.shape[0], hd_img.shape[1], 3,
                                                                   device=pydiffvg.get_device()) * (
                             1 - hd_img[:, :, 3:4])
        pydiffvg.imwrite(hd_img.cpu(), os.path.join(exp_dir, str(t) + '.png'), gamma=gamma)

        img = img[:, :, :3]
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2)  # NHWC -> NCHW

        NUM_AUGS = num_augs

        # img_augs = []
        # for n in range(NUM_AUGS):
        #     img_augs.append(augment_trans(img))
        # im_batch = torch.cat(img_augs)

        img_augs = [img] * NUM_AUGS
        img_augs = torch.cat(img_augs).to('cuda')
        im_batch = augment_trans(img_augs)

        image_features = clip_model.encode_image(im_batch)
        # for n in range(NUM_AUGS):
        #     loss -= torch.cosine_similarity(text_features, image_features[n:n+1], dim=1)
        #     if use_negative:
        #         loss += torch.cosine_similarity(text_features_neg1, image_features[n:n+1], dim=1) * 0.3
        #         loss += torch.cosine_similarity(text_features_neg2, image_features[n:n+1], dim=1) * 0.3
        # if use_negative:
        #     loss += torch.cosine_similarity(text_features_neg1, image_features[n:n+1], dim=1) * 0.3
        #     loss += torch.cosine_similarity(text_features_neg2, image_features[n:n+1], dim=1) * 0.3

        loss = -torch.cosine_similarity(text_features, image_features, dim=1).sum() / num_augs

        # Backpropagate the gradients.
        loss.backward()

        # Take a gradient descent step.
        points_optim.step()
        width_optim.step()
        color_optim.step()
        for path in shapes:
            path.stroke_width.data.clamp_(1.0, max_width)
        for group in shape_groups:
            group.stroke_color.data.clamp_(0.0, 1.0)

        if t % 10 == 0:
            show_img(img.detach().cpu().numpy()[0])
            # show_img(torch.cat([img.detach(), img_aug.detach()], axis=3).cpu().numpy()[0])
            print('render loss:', loss.item())
            print('iteration:', t)
            with torch.no_grad():
                im_norm = image_features / image_features.norm(dim=-1, keepdim=True)
                noun_norm = nouns_features / nouns_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * im_norm @ noun_norm.T).softmax(dim=-1)
                values, indices = similarity[0].topk(5)
                print("\nTop predictions:\n")
                for value, index in zip(values, indices):
                    print(f"{nouns[index]:>16s}: {100 * value.item():.2f}%")

    fnames = [os.path.join(exp_dir, f'{t}.png') for t in range(num_iter)]
    images = [Image.open(f) for f in fnames]
    return images