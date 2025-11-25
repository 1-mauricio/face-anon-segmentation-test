"""
Script para visualizar as máscaras de segmentação facial.

Este script demonstra como usar a segmentação facial para criar máscaras
de diferentes partes do rosto e visualizá-las sobrepostas na imagem original.
"""

import cv2
import numpy as np
from PIL import Image
import face_alignment
from .segmentation import get_mask_from_landmarks, visualize_mask, get_segmented_regions
from .extractor import get_transform_mat, FaceType
from diffusers.utils import load_image


def visualize_all_segments(image_path, output_path=None, face_image_size=512):
    """
    Visualiza todas as regiões segmentadas disponíveis em uma imagem.
    
    Args:
        image_path (str): Caminho para a imagem de entrada.
        output_path (str, optional): Caminho para salvar a visualização.
        face_image_size (int): Tamanho da face alinhada.
    
    Returns:
        PIL.Image: Imagem com todas as máscaras visualizadas.
    """
    # Carregar imagem
    image = load_image(image_path)
    image_np = np.array(image)
    
    # Garantir RGB
    if image_np.shape[2] == 4:
        image_np = image_np[:, :, :3]
    
    # Inicializar face alignment
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D, 
        face_detector="sfd", 
        device=device
    )
    
    # Detectar landmarks
    preds = fa.get_landmarks(image_np)
    
    if preds is None:
        print("Nenhum rosto detectado na imagem.")
        return image
    
    result_np = image_np.copy()
    
    for landmarks in preds:
        # Obter matriz de transformação
        mat = get_transform_mat(landmarks, face_image_size, FaceType.WHOLE_FACE)
        
        # Extrair face alinhada
        face_aligned = cv2.warpAffine(
            result_np,
            mat,
            (face_image_size, face_image_size),
            cv2.INTER_LANCZOS4,
            borderValue=(255, 255, 255),
        )
        
        # Transformar landmarks para o espaço alinhado
        pts = np.array([landmarks], dtype=np.float32)
        aligned_landmarks = cv2.transform(pts, mat)[0]
        
        # Obter todas as regiões segmentadas
        masks = get_segmented_regions(aligned_landmarks, (face_image_size, face_image_size))
        
        # Visualizar cada máscara
        for region_name, mask in masks.items():
            if mask.sum() > 0:  # Se a máscara não estiver vazia
                # Transformar máscara de volta para o espaço original
                mask_resized = cv2.warpAffine(
                    mask,
                    cv2.invertAffineTransform(mat),
                    (result_np.shape[1], result_np.shape[0]),
                    flags=cv2.INTER_NEAREST,
                    borderValue=0
                )
                
                # Visualizar máscara
                overlay = visualize_mask(result_np, mask_resized, alpha=0.4)
                result_np = overlay
    
    result_image = Image.fromarray(result_np)
    
    if output_path:
        result_image.save(output_path)
        print(f"Visualização salva em: {output_path}")
    
    return result_image


def visualize_specific_segments(image_path, features, output_path=None, face_image_size=512):
    """
    Visualiza máscaras específicas de segmentação.
    
    Args:
        image_path (str): Caminho para a imagem de entrada.
        features (list): Lista de características para visualizar.
        output_path (str, optional): Caminho para salvar a visualização.
        face_image_size (int): Tamanho da face alinhada.
    
    Returns:
        PIL.Image: Imagem com as máscaras visualizadas.
    """
    # Carregar imagem
    image = load_image(image_path)
    image_np = np.array(image)
    
    # Garantir RGB
    if image_np.shape[2] == 4:
        image_np = image_np[:, :, :3]
    
    # Inicializar face alignment
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D, 
        face_detector="sfd", 
        device=device
    )
    
    # Detectar landmarks
    preds = fa.get_landmarks(image_np)
    
    if preds is None:
        print("Nenhum rosto detectado na imagem.")
        return image
    
    result_np = image_np.copy()
    
    for landmarks in preds:
        # Obter matriz de transformação
        mat = get_transform_mat(landmarks, face_image_size, FaceType.WHOLE_FACE)
        
        # Extrair face alinhada
        face_aligned = cv2.warpAffine(
            result_np,
            mat,
            (face_image_size, face_image_size),
            cv2.INTER_LANCZOS4,
            borderValue=(255, 255, 255),
        )
        
        # Transformar landmarks para o espaço alinhado
        pts = np.array([landmarks], dtype=np.float32)
        aligned_landmarks = cv2.transform(pts, mat)[0]
        
        # Gerar máscara para as características especificadas
        mask = get_mask_from_landmarks(
            aligned_landmarks, 
            (face_image_size, face_image_size), 
            features,
            dilate_radius=3,
            smooth_edges=True
        )
        
        # Transformar máscara de volta para o espaço original
        mask_resized = cv2.warpAffine(
            mask,
            cv2.invertAffineTransform(mat),
            (result_np.shape[1], result_np.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderValue=0
        )
        
        # Visualizar máscara
        overlay = visualize_mask(result_np, mask_resized, alpha=0.5)
        result_np = overlay
    
    result_image = Image.fromarray(result_np)
    
    if output_path:
        result_image.save(output_path)
        print(f"Visualização salva em: {output_path}")
    
    return result_image


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualizar segmentação facial")
    parser.add_argument("--image_path", type=str, required=True, help="Caminho para a imagem")
    parser.add_argument("--output_path", type=str, default=None, help="Caminho para salvar a visualização")
    parser.add_argument("--features", type=str, nargs="+", default=None, 
                       help="Características para visualizar (ex: eyes mouth nose)")
    parser.add_argument("--all", action="store_true", help="Visualizar todas as regiões")
    
    args = parser.parse_args()
    
    if args.all:
        visualize_all_segments(args.image_path, args.output_path)
    elif args.features:
        visualize_specific_segments(args.image_path, args.features, args.output_path)
    else:
        # Exemplo padrão: visualizar olhos e boca
        visualize_specific_segments(args.image_path, ['eyes', 'mouth'], args.output_path)

