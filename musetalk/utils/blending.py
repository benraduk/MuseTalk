from PIL import Image, ImageDraw
import numpy as np
import cv2
import copy


def get_crop_box(box, expand):
    x, y, x1, y1 = box
    x_c, y_c = (x+x1)//2, (y+y1)//2
    w, h = x1-x, y1-y
    s = int(max(w, h)//2*expand)
    crop_box = [x_c-s, y_c-s, x_c+s, y_c+s]
    return crop_box, s


def face_seg(image, mode="raw", fp=None):
    """
    对图像进行面部解析，生成面部区域的掩码。

    Args:
        image (PIL.Image): 输入图像。

    Returns:
        PIL.Image: 面部区域的掩码图像。
    """
    seg_image = fp(image, mode=mode)  # 使用 FaceParsing 模型解析面部
    if seg_image is None:
        print("error, no person_segment")  # 如果没有检测到面部，返回错误
        return None

    seg_image = seg_image.resize(image.size)  # 将掩码图像调整为输入图像的大小
    return seg_image


def get_image(image, face, face_box, upper_boundary_ratio=0.5, expand=1.5, mode="raw", fp=None, use_elliptical_mask=True, ellipse_padding_factor=0.1, blur_kernel_ratio=0.05, landmarks=None, mouth_vertical_offset=0.0):
    """
    将裁剪的面部图像粘贴回原始图像，并进行一些处理。
    Enhanced with landmark-based surgical mouth positioning for improved accuracy.

    Args:
        image (numpy.ndarray): 原始图像（身体部分）。
        face (numpy.ndarray): 裁剪的面部图像。
        face_box (tuple): 面部边界框的坐标 (x, y, x1, y1)。
        upper_boundary_ratio (float): 用于控制面部区域的保留比例。
        expand (float): 扩展因子，用于放大裁剪框。
        mode: 融合mask构建方式 
        use_elliptical_mask (bool): 是否使用椭圆形掩码而不是矩形掩码。
        ellipse_padding_factor (float): 椭圆掩码的内边距因子，控制椭圆相对于面部边界的大小。
        blur_kernel_ratio (float): 高斯模糊核大小比例，用于平滑掩码边缘。
        landmarks (list): YOLOv8 facial landmarks [(left_eye), (right_eye), (nose), (left_mouth), (right_mouth)]
        mouth_vertical_offset (float): Vertical offset for mouth positioning (positive = lower, negative = higher)

    Returns:
        numpy.ndarray: 处理后的图像。
    """
    # 将 numpy 数组转换为 PIL 图像
    body = Image.fromarray(image[:, :, ::-1])  # 身体部分图像(整张图)
    face = Image.fromarray(face[:, :, ::-1])  # 面部图像

    x, y, x1, y1 = face_box  # 获取面部边界框的坐标
    crop_box, s = get_crop_box(face_box, expand)  # 计算扩展后的裁剪框
    x_s, y_s, x_e, y_e = crop_box  # 裁剪框的坐标
    face_position = (x, y)  # 面部在原始图像中的位置

    # 从身体图像中裁剪出扩展后的面部区域（下巴到边界有距离）
    face_large = body.crop(crop_box)
        
    ori_shape = face_large.size  # 裁剪后图像的原始尺寸

    # 对裁剪后的面部区域进行面部解析，生成掩码
    mask_image = face_seg(face_large, mode=mode, fp=fp)
    
    mask_small = mask_image.crop((x - x_s, y - y_s, x1 - x_s, y1 - y_s))  # 裁剪出面部区域的掩码
    
    # Create mask with surgical precision using landmarks if available
    mask_image = Image.new('L', ori_shape, 0)  # 创建一个全黑的掩码图像
    
    if landmarks is not None and len(landmarks) >= 5:
        # 🎯 SURGICAL POSITIONING: Use YOLOv8 landmarks for precise mouth region
        left_eye, right_eye, nose_tip, left_mouth, right_mouth = landmarks
        
        # Calculate mouth-specific region for surgical precision
        mouth_center_x = (left_mouth[0] + right_mouth[0]) / 2
        mouth_center_y = (left_mouth[1] + right_mouth[1]) / 2
        mouth_width = abs(right_mouth[0] - left_mouth[0])
        
        # Enhanced mouth corner analysis for better positioning
        mouth_corner_height_diff = abs(left_mouth[1] - right_mouth[1])
        mouth_angle = mouth_corner_height_diff / max(mouth_width, 1)  # Prevent division by zero
        
        # Calculate nose-to-mouth distance for proportional sizing
        nose_to_mouth_dist = abs(nose_tip[1] - mouth_center_y)
        
        # Create landmark-based elliptical mask focused on mouth region
        face_width = x1 - x
        face_height = y1 - y
        
        # Convert global landmarks to local face coordinates
        local_mouth_center_x = mouth_center_x - x
        local_mouth_center_y = mouth_center_y - y
        local_left_mouth_x = left_mouth[0] - x
        local_right_mouth_x = right_mouth[0] - x
        local_mouth_y = left_mouth[1] - y  # Use left mouth Y (they should be similar)
        
        # Apply vertical offset for fine-tuning mouth position
        # Positive offset moves mouth down, negative moves it up
        offset_pixels = mouth_vertical_offset * face_height  # Convert ratio to pixels
        local_mouth_center_y += offset_pixels
        local_mouth_y += offset_pixels
        
        # Create surgical mouth mask
        mouth_mask = Image.new('L', (face_width, face_height), 0)
        draw = ImageDraw.Draw(mouth_mask)
        
        # Calculate surgical mouth region with landmark precision
        mouth_region_width = mouth_width * (1.0 + ellipse_padding_factor * 2)  # Precise width based on actual mouth
        mouth_region_height = nose_to_mouth_dist * 0.8  # Height based on nose-mouth proportion
        
        # Ensure minimum size for small faces
        mouth_region_width = max(mouth_region_width, face_width * 0.3)
        mouth_region_height = max(mouth_region_height, face_height * 0.2)
        
        # Calculate ellipse bounds centered on actual mouth position
        ellipse_left = local_mouth_center_x - mouth_region_width / 2
        ellipse_top = local_mouth_center_y - mouth_region_height / 2
        ellipse_right = local_mouth_center_x + mouth_region_width / 2
        ellipse_bottom = local_mouth_center_y + mouth_region_height / 2
        
        # Ensure ellipse stays within face bounds
        ellipse_left = max(0, ellipse_left)
        ellipse_top = max(0, ellipse_top)
        ellipse_right = min(face_width, ellipse_right)
        ellipse_bottom = min(face_height, ellipse_bottom)
        
        # Draw surgical mouth ellipse
        draw.ellipse([ellipse_left, ellipse_top, ellipse_right, ellipse_bottom], fill=255)
        
        # Apply face parsing mask for additional refinement
        mouth_array = np.array(mouth_mask)
        mask_small_array = np.array(mask_small)
        combined_mask = np.minimum(mouth_array, mask_small_array)
        final_face_mask = Image.fromarray(combined_mask)
        
        # Paste the surgical landmark-based mask
        mask_image.paste(final_face_mask, (x - x_s, y - y_s))
        
        offset_info = f", offset {mouth_vertical_offset:+.2f}" if mouth_vertical_offset != 0.0 else ""
        print(f"🎯 Surgical positioning: mouth center ({mouth_center_x:.1f}, {mouth_center_y + offset_pixels:.1f}), width {mouth_width:.1f}px{offset_info}")
        
    elif use_elliptical_mask:
        # Fallback: Create elliptical mask for more natural blending (original method)
        face_width = x1 - x
        face_height = y1 - y
        
        # Create elliptical mask for the face region
        ellipse_mask = Image.new('L', (face_width, face_height), 0)
        draw = ImageDraw.Draw(ellipse_mask)
        
        # Calculate padding to make ellipse smaller than face bounds
        padding_w = int(face_width * ellipse_padding_factor)
        padding_h = int(face_height * ellipse_padding_factor)
        
        # Draw ellipse (white = include area, black = exclude)
        draw.ellipse([padding_w, padding_h, face_width - padding_w, face_height - padding_h], fill=255)
        
        # Apply the face parsing mask to the elliptical mask (intersection)
        ellipse_array = np.array(ellipse_mask)
        mask_small_array = np.array(mask_small)
        combined_mask = np.minimum(ellipse_array, mask_small_array)
        final_face_mask = Image.fromarray(combined_mask)
        
        # Paste the combined elliptical + parsing mask
        mask_image.paste(final_face_mask, (x - x_s, y - y_s))
        
        print(f"🔧 Fallback: elliptical mask (no landmarks available)")
    else:
        # Original rectangular mask behavior
        mask_image.paste(mask_small, (x - x_s, y - y_s, x1 - x_s, y1 - y_s))  # 将面部掩码粘贴到全黑图像上
        print(f"📦 Basic: rectangular mask")
    
    
    # 保留面部区域的上半部分（用于控制说话区域）
    width, height = mask_image.size
    top_boundary = int(height * upper_boundary_ratio)  # 计算上半部分的边界
    modified_mask_image = Image.new('L', ori_shape, 0)  # 创建一个新的全黑掩码图像
    modified_mask_image.paste(mask_image.crop((0, top_boundary, width, height)), (0, top_boundary))  # 粘贴上半部分掩码
    
    
    # 对掩码进行高斯模糊，使边缘更平滑
    blur_kernel_size = int(blur_kernel_ratio * ori_shape[0] // 2 * 2) + 1  # 计算模糊核大小
    mask_array = cv2.GaussianBlur(np.array(modified_mask_image), (blur_kernel_size, blur_kernel_size), 0)  # 高斯模糊
    #mask_array = np.array(modified_mask_image)
    mask_image = Image.fromarray(mask_array)  # 将模糊后的掩码转换回 PIL 图像
    
    # 将裁剪的面部图像粘贴回扩展后的面部区域
    face_large.paste(face, (x - x_s, y - y_s, x1 - x_s, y1 - y_s))
    
    body.paste(face_large, crop_box[:2], mask_image)
    
    body = np.array(body)  # 将 PIL 图像转换回 numpy 数组

    return body[:, :, ::-1]  # 返回处理后的图像（BGR 转 RGB）


def get_image_blending(image, face, face_box, mask_array, crop_box):
    body = Image.fromarray(image[:,:,::-1])
    face = Image.fromarray(face[:,:,::-1])

    x, y, x1, y1 = face_box
    x_s, y_s, x_e, y_e = crop_box
    face_large = body.crop(crop_box)

    mask_image = Image.fromarray(mask_array)
    mask_image = mask_image.convert("L")
    face_large.paste(face, (x-x_s, y-y_s, x1-x_s, y1-y_s))
    body.paste(face_large, crop_box[:2], mask_image)
    body = np.array(body)
    return body[:,:,::-1]


def get_image_prepare_material(image, face_box, upper_boundary_ratio=0.5, expand=1.5, fp=None, mode="raw"):
    body = Image.fromarray(image[:,:,::-1])

    x, y, x1, y1 = face_box
    #print(x1-x,y1-y)
    crop_box, s = get_crop_box(face_box, expand)
    x_s, y_s, x_e, y_e = crop_box

    face_large = body.crop(crop_box)
    ori_shape = face_large.size

    mask_image = face_seg(face_large, mode=mode, fp=fp)
    mask_small = mask_image.crop((x-x_s, y-y_s, x1-x_s, y1-y_s))
    mask_image = Image.new('L', ori_shape, 0)
    mask_image.paste(mask_small, (x-x_s, y-y_s, x1-x_s, y1-y_s))

    # keep upper_boundary_ratio of talking area
    width, height = mask_image.size
    top_boundary = int(height * upper_boundary_ratio)
    modified_mask_image = Image.new('L', ori_shape, 0)
    modified_mask_image.paste(mask_image.crop((0, top_boundary, width, height)), (0, top_boundary))

    blur_kernel_size = int(0.1 * ori_shape[0] // 2 * 2) + 1
    mask_array = cv2.GaussianBlur(np.array(modified_mask_image), (blur_kernel_size, blur_kernel_size), 0)
    return mask_array, crop_box
