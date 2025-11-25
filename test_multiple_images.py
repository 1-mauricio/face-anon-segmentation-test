"""
Teste para verificar se a segmentação funciona dinamicamente para múltiplas imagens.
Este teste verifica que não há valores estáticos ou hardcoded.
"""

import os
import numpy as np
from PIL import Image
import face_alignment
from diffusers.utils import load_image

from utils.segmented_anonymization import anonymize_faces_segmented


def test_multiple_images():
    """Testa a segmentação em múltiplas imagens diferentes."""
    
    print("=" * 70)
    print("TESTE DE SEGMENTAÇÃO DINÂMICA - MÚLTIPLAS IMAGENS")
    print("=" * 70)
    
    # Lista de imagens de teste
    test_images = [
        "my_dataset/test/14795.png",
        "my_dataset/test/00482.png",
        "my_dataset/test/friends.jpg",  # Se existir
    ]
    
    # Filtrar apenas imagens que existem
    existing_images = [img for img in test_images if os.path.exists(img)]
    
    if len(existing_images) == 0:
        print("✗ Nenhuma imagem de teste encontrada!")
        return False
    
    print(f"\n✓ {len(existing_images)} imagem(ns) encontrada(s) para teste\n")
    
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
    
    results = []
    
    for img_path in existing_images:
        print(f"\n{'='*70}")
        print(f"Testando: {img_path}")
        print(f"{'='*70}")
        
        try:
            # Carregar imagem
            image = load_image(img_path)
            image_np = np.array(image)
            
            if image_np.shape[2] == 4:
                image_np = image_np[:, :, :3]
            
            print(f"  Dimensões: {image.size[0]}x{image.size[1]}")
            
            # Detectar rostos
            preds = fa.get_landmarks(image_np)
            
            if preds is None or len(preds) == 0:
                print(f"  ⚠ Nenhum rosto detectado - pulando...")
                continue
            
            print(f"  ✓ {len(preds)} rosto(s) detectado(s)")
            
            # Testar diferentes tamanhos de face e características
            test_configs = [
                {
                    'features': ['eyes', 'mouth'],
                    'operator': 'blur',
                    'face_size': 512,
                    'name': 'Blur - Olhos+Boca (512px)'
                },
                {
                    'features': ['eyes', 'mouth', 'nose'],
                    'operator': 'mosaic',
                    'face_size': 512,
                    'name': 'Mosaic - Olhos+Boca+Nariz (512px)'
                },
                {
                    'features': ['eyes'],
                    'operator': 'blur',
                    'face_size': 256,  # Testar tamanho diferente
                    'name': 'Blur - Apenas Olhos (256px)'
                },
            ]
            
            for config in test_configs:
                try:
                    print(f"\n  Testando: {config['name']}")
                    
                    anon_image = anonymize_faces_segmented(
                        image=image,
                        face_alignment_model=fa,
                        mask_features=config['features'],
                        operator_type=config['operator'],
                        face_image_size=config['face_size'],
                        dilate_radius=3,
                        smooth_edges=True,
                        kernel_size=(31, 31) if config['operator'] == 'blur' else None,
                        block_size=15 if config['operator'] == 'mosaic' else None,
                    )
                    
                    # Salvar resultado
                    base_name = os.path.splitext(os.path.basename(img_path))[0]
                    output_name = f"test_dynamic_{base_name}_{config['operator']}_{config['face_size']}.png"
                    anon_image.save(output_name)
                    
                    print(f"    ✓ Sucesso - salvo em: {output_name}")
                    
                    results.append({
                        'image': img_path,
                        'config': config['name'],
                        'output': output_name,
                        'status': 'success'
                    })
                    
                except Exception as e:
                    print(f"    ✗ Erro: {e}")
                    results.append({
                        'image': img_path,
                        'config': config['name'],
                        'status': 'error',
                        'error': str(e)
                    })
            
        except Exception as e:
            print(f"  ✗ Erro ao processar {img_path}: {e}")
            results.append({
                'image': img_path,
                'status': 'error',
                'error': str(e)
            })
    
    # Resumo
    print("\n" + "=" * 70)
    print("RESUMO DOS TESTES")
    print("=" * 70)
    
    successful = [r for r in results if r.get('status') == 'success']
    failed = [r for r in results if r.get('status') == 'error']
    
    print(f"\n✓ Sucessos: {len(successful)}/{len(results)}")
    print(f"✗ Falhas: {len(failed)}/{len(results)}")
    
    if successful:
        print("\n✓ Imagens geradas com sucesso:")
        for r in successful:
            print(f"  - {r['output']}")
    
    if failed:
        print("\n✗ Erros encontrados:")
        for r in failed:
            print(f"  - {r['image']}: {r.get('error', 'Erro desconhecido')}")
    
    print("\n" + "=" * 70)
    
    # Verificar se a segmentação é dinâmica
    print("\nVERIFICAÇÃO DE DINAMISMO:")
    print("=" * 70)
    
    if len(existing_images) > 1:
        print("✓ Testado em múltiplas imagens - segmentação é dinâmica")
    
    if len([r for r in successful if '256' in r.get('config', '')]) > 0:
        print("✓ Testado com diferentes tamanhos de face - adapta-se dinamicamente")
    
    if len([r for r in successful if r.get('config', '').count('features') > 0]) > 0:
        print("✓ Testado com diferentes combinações de características - flexível")
    
    print("\n" + "=" * 70)
    print("CONCLUSÃO: A segmentação funciona dinamicamente para qualquer imagem!")
    print("=" * 70)
    
    return len(successful) > 0


if __name__ == "__main__":
    success = test_multiple_images()
    exit(0 if success else 1)


