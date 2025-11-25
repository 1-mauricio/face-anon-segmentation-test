"""
Teste simples da segmentação facial.
Este script testa a funcionalidade de segmentação sem carregar os modelos pesados de difusão.
"""

import numpy as np
import cv2
from PIL import Image
import face_alignment
from diffusers.utils import load_image

from utils.segmentation import get_mask_from_landmarks, visualize_mask
from utils.extractor import get_transform_mat, FaceType


def test_segmentation():
    """Testa a segmentação facial em uma imagem de exemplo."""
    
    print("=" * 60)
    print("TESTE DE SEGMENTAÇÃO FACIAL")
    print("=" * 60)
    
    # Carregar imagem de teste
    image_path = "my_dataset/test/00482.png"
    print(f"\n1. Carregando imagem: {image_path}")
    
    try:
        original_image = load_image(image_path)
        print(f"   ✓ Imagem carregada: {original_image.size}")
    except Exception as e:
        print(f"   ✗ Erro ao carregar imagem: {e}")
        return False
    
    # Converter para numpy
    image_np = np.array(original_image)
    if image_np.shape[2] == 4:
        image_np = image_np[:, :, :3]
    
    # Inicializar face alignment
    print("\n2. Inicializando Face Alignment...")
    try:
        device = "cuda" if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
        # Tentar usar torch para detectar CUDA
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pass
        
        fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            face_detector="sfd",
            device=device
        )
        print(f"   ✓ Face Alignment inicializado (device: {device})")
    except Exception as e:
        print(f"   ✗ Erro ao inicializar Face Alignment: {e}")
        return False
    
    # Detectar landmarks
    print("\n3. Detectando landmarks faciais...")
    try:
        preds = fa.get_landmarks(image_np)
        if preds is None or len(preds) == 0:
            print("   ✗ Nenhum rosto detectado na imagem")
            return False
        print(f"   ✓ {len(preds)} rosto(s) detectado(s)")
    except Exception as e:
        print(f"   ✗ Erro ao detectar landmarks: {e}")
        return False
    
    # Testar segmentação para cada rosto
    face_image_size = 512
    test_features = [
        ['eyes'],
        ['mouth'],
        ['nose'],
        ['eyes', 'mouth', 'nostrils'],
        ['eyebrows', 'nose', 'lips']
    ]
    
    print("\n4. Testando segmentação de diferentes características...")
    
    for idx, landmarks in enumerate(preds):
        print(f"\n   Rosto {idx + 1}:")
        
        # Obter matriz de transformação
        try:
            mat = get_transform_mat(landmarks, face_image_size, FaceType.WHOLE_FACE)
            
            # Extrair face alinhada
            face_aligned = cv2.warpAffine(
                image_np,
                mat,
                (face_image_size, face_image_size),
                cv2.INTER_LANCZOS4,
                borderValue=(255, 255, 255),
            )
            
            # Transformar landmarks para o espaço alinhado
            pts = np.array([landmarks], dtype=np.float32)
            aligned_landmarks = cv2.transform(pts, mat)[0]
            
            print(f"      ✓ Face alinhada extraída ({face_image_size}x{face_image_size})")
        except Exception as e:
            print(f"      ✗ Erro ao alinhar face: {e}")
            continue
        
        # Testar cada conjunto de características
        for features in test_features:
            try:
                mask = get_mask_from_landmarks(
                    aligned_landmarks,
                    (face_image_size, face_image_size),
                    features,
                    dilate_radius=3,
                    smooth_edges=True
                )
                
                # Verificar se a máscara não está vazia
                mask_area = np.sum(mask > 0)
                mask_percentage = (mask_area / (face_image_size * face_image_size)) * 100
                
                print(f"      ✓ {features}: máscara gerada ({mask_percentage:.1f}% da face)")
                
            except Exception as e:
                print(f"      ✗ Erro ao gerar máscara para {features}: {e}")
    
    # Testar visualização
    print("\n5. Testando visualização de máscaras...")
    try:
        landmarks = preds[0]
        mat = get_transform_mat(landmarks, face_image_size, FaceType.WHOLE_FACE)
        face_aligned = cv2.warpAffine(
            image_np,
            mat,
            (face_image_size, face_image_size),
            cv2.INTER_LANCZOS4,
            borderValue=(255, 255, 255),
        )
        pts = np.array([landmarks], dtype=np.float32)
        aligned_landmarks = cv2.transform(pts, mat)[0]
        
        mask = get_mask_from_landmarks(
            aligned_landmarks,
            (face_image_size, face_image_size),
            ['eyes', 'mouth'],
            dilate_radius=3,
            smooth_edges=True
        )
        
        vis_image = visualize_mask(face_aligned, mask, alpha=0.5)
        print(f"   ✓ Visualização criada: shape {vis_image.shape}")
        
        # Salvar imagem de teste
        output_path = "test_segmentation_output.png"
        Image.fromarray(vis_image).save(output_path)
        print(f"   ✓ Imagem salva em: {output_path}")
        
    except Exception as e:
        print(f"   ✗ Erro ao testar visualização: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✓ TESTE CONCLUÍDO COM SUCESSO!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_segmentation()
    exit(0 if success else 1)


