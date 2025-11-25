import cv2
import numpy as np
from PIL import Image

def apply_blur(image, mask, kernel_size=(21, 21), **kwargs):
    """
    Applies Gaussian blur to the masked region.
    
    Args:
        image (np.ndarray): Input image (H, W, 3).
        mask (np.ndarray): Binary mask (H, W).
        kernel_size (tuple): Gaussian kernel size.
        kwargs: Ignored extra arguments.
    
    Returns:
        np.ndarray: Processed image.
    """
    blurred = cv2.GaussianBlur(image, kernel_size, 0)
    # Expand mask to 3 channels
    mask_3d = mask[:, :, None] / 255.0
    
    # Blend
    output = image * (1 - mask_3d) + blurred * mask_3d
    return output.astype(np.uint8)

def apply_mosaic(image, mask, block_size=10, **kwargs):
    """
    Applies mosaic (pixelation) to the masked region.
    
    Args:
        image (np.ndarray): Input image (H, W, 3).
        mask (np.ndarray): Binary mask (H, W).
        block_size (int): Size of the pixel blocks.
        kwargs: Ignored extra arguments.
    
    Returns:
        np.ndarray: Processed image.
    """
    h, w = image.shape[:2]
    # Resize down
    small = cv2.resize(image, (w // block_size, h // block_size), interpolation=cv2.INTER_LINEAR)
    # Resize up
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    
    mask_3d = mask[:, :, None] / 255.0
    output = image * (1 - mask_3d) + pixelated * mask_3d
    return output.astype(np.uint8)

def apply_diffusion(image, mask, pipe, generator=None, **kwargs):
    """
    Aplica anonimização baseada em difusão (face_anon_simple) apenas na região segmentada.
    
    Esta função usa o pipeline de difusão para anonimizar a face inteira, mas aplica
    o resultado apenas nas regiões especificadas pela máscara. Isso permite anonimização
    seletiva de características faciais específicas (olhos, boca, nariz, etc.).
    
    Args:
        image (np.ndarray): Imagem da face alinhada (H, W, 3).
        mask (np.ndarray): Máscara binária (H, W) indicando a região a anonimizar.
        pipe: Pipeline de difusão (StableDiffusionReferenceNetPipeline).
        generator: Gerador PyTorch para reprodutibilidade.
        kwargs: Argumentos adicionais para o pipeline:
                - num_inference_steps (int): Número de passos de inferência (padrão: 50)
                - guidance_scale (float): Escala de orientação (padrão: 4.0)
                - anonymization_degree (float): Grau de anonimização (padrão: 1.25)
                - width (int): Largura da imagem (padrão: tamanho da imagem)
                - height (int): Altura da imagem (padrão: tamanho da imagem)
    
    Returns:
        np.ndarray: Imagem processada com anonimização aplicada apenas na região da máscara.
    """
    # Converter numpy para PIL para o pipeline
    pil_image = Image.fromarray(image)
    
    # Garantir que width e height estão nos kwargs se não especificados
    if 'width' not in kwargs:
        kwargs['width'] = image.shape[1]
    if 'height' not in kwargs:
        kwargs['height'] = image.shape[0]
    
    # Executar o pipeline para obter a face anonimizada completa
    # O pipeline precisa da face inteira para funcionar corretamente
    anon_pil = pipe(
        source_image=pil_image,
        conditioning_image=pil_image,
        generator=generator,
        **kwargs
    ).images[0]
    
    anon_array = np.array(anon_pil)
    
    # Redimensionar se necessário (o pipeline pode redimensionar)
    if anon_array.shape != image.shape:
        anon_array = cv2.resize(
            anon_array, 
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_LANCZOS4
        )
    
    # Aplicar máscara: apenas a região segmentada recebe a anonimização
    # A máscara é normalizada para [0, 1] e expandida para 3 canais
    mask_3d = mask[:, :, None] / 255.0
    
    # Combinar: manter original fora da máscara, usar anonimizado dentro da máscara
    output = image * (1 - mask_3d) + anon_array * mask_3d
    
    return output.astype(np.uint8)
