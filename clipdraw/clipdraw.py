from .helpers import *
from .nouns import noun_prompts, nouns

import clip
import torch
from torchvision import transforms
import pydiffvg
import random

def clip_draw_render(model, prompt="Watercolor painting of an underwater submarine.",
                     num_paths=256, num_iter=100, max_width=50, use_normalized_clip=True,
                     exp_size=600, n_save=1, exp_dir='clipdraw_out'):
    device = torch.device('cuda:0')

    nouns_features = model.encode_text(torch.cat([clip.tokenize(noun_prompts).to(device)]))

    text_input = clip.tokenize(prompt).to(device)

    # Calculate features
    with torch.no_grad():
        text_features = model.encode_text(text_input)

    pydiffvg.set_print_timing(False)

    gamma = 1.0

    # Use GPU if available
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    pydiffvg.set_device(device)

    canvas_width, canvas_height = 224, 224

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
    shapes = []
    shape_groups = []
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
        shapes.append(path)
        path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1]), fill_color=None,
                                         stroke_color=torch.tensor(
                                             [random.random(), random.random(), random.random(), random.random()]))
        shape_groups.append(path_group)

    # Just some diffvg setup
    scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, canvas_height, shapes, shape_groups)
    render = pydiffvg.RenderFunction.apply
    img = render(canvas_width, canvas_height, 2, 2, 0, None, *scene_args)
    points_vars = []
    stroke_width_vars = []
    color_vars = []
    for path in shapes:
        path.points.requires_grad = True
        points_vars.append(path.points)
        path.stroke_width.requires_grad = True
        stroke_width_vars.append(path.stroke_width)
    for group in shape_groups:
        group.stroke_color.requires_grad = True
        color_vars.append(group.stroke_color)

    # Optimizers
    points_optim = torch.optim.Adam(points_vars, lr=1.0)
    width_optim = torch.optim.Adam(stroke_width_vars, lr=0.1)
    color_optim = torch.optim.Adam(color_vars, lr=0.01)

    # Run the main optimization loop
    for t in range(num_iter):

        # Anneal learning rate (makes videos look cleaner)
        if t == int(num_iter * 0.5):
            for g in points_optim.param_groups:
                g['lr'] = 0.4
        if t == int(num_iter * 0.75):
            for g in points_optim.param_groups:
                g['lr'] = 0.1

        points_optim.zero_grad()
        width_optim.zero_grad()
        color_optim.zero_grad()
        scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, canvas_height, shapes, shape_groups)
        img = render(canvas_width, canvas_height, 2, 2, t, None, *scene_args)
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3,
                                                          device=pydiffvg.get_device()) * (1 - img[:, :, 3:4])
        if t % n_save == 0:
            exp_img = render(exp_size, exp_size, 2, 2, t, None, *scene_args)
            exp_img = exp_img[:, :, 3:4] * exp_img[:, :, :3] + torch.ones(exp_img.shape[0], exp_img.shape[1], 3,
                                                                          device=pydiffvg.get_device()) * (
                                  1 - exp_img[:, :, 3:4])
            save_dir = os.path.join(exp_dir, 'iter_{}.png'.format(int(t / n_save)))
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
        image_features = model.encode_image(im_batch)
        for n in range(NUM_AUGS):
            loss -= torch.cosine_similarity(text_features, image_features[n:n + 1], dim=1)

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

        if t == num_iter - 1:
            save_dir = os.path.join(exp_dir, 'final.png')
            pydiffvg.imwrite(exp_img.cpu(), save_dir, gamma=gamma)
