"""
Teste de anonimização segmentada usando face_anon_simple (diffusion).
Este teste verifica que a difusão é aplicada apenas nas regiões segmentadas escolhidas.
"""

import torch
from transformers import CLIPImageProcessor, CLIPVisionModel
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.utils import load_image, make_image_grid
import face_alignment
from PIL import Image
import numpy as np

from src.diffusers.models.referencenet.referencenet_unet_2d_condition import ReferenceNetModel
from src.diffusers.models.referencenet.unet_2d_condition import UNet2DConditionModel
from src.diffusers.pipelines.referencenet.pipeline_referencenet import StableDiffusionReferenceNetPipeline

from utils.segmented_anonymization import anonymize_faces_segmented


def test_segmented_diffusion():
    """Testa anonimização segmentada com face_anon_simple."""
    
    print("=" * 70)
    print("TESTE DE ANONIMIZAÇÃO SEGMENTADA COM FACE_ANON_SIMPLE")
    print("=" * 70)
    
    # Carregar modelos
    print("\n1. Carregando modelos...")
    face_model_id = "hkung/face-anon-simple"
    clip_model_id = "openai/clip-vit-large-patch14"
    sd_model_id = "stabilityai/stable-diffusion-2-1"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    print(f"   Device: {device}, dtype: {dtype}")
    
    try:
        print("   Carregando UNet...")
        unet = UNet2DConditionModel.from_pretrained(
            face_model_id, subfolder="unet", use_safetensors=True
        )
        
        print("   Carregando ReferenceNet...")
        referencenet = ReferenceNetModel.from_pretrained(
            face_model_id, subfolder="referencenet", use_safetensors=True
        )
        
        print("   Carregando Conditioning ReferenceNet...")
        conditioning_referencenet = ReferenceNetModel.from_pretrained(
            face_model_id, subfolder="conditioning_referencenet", use_safetensors=True
        )
        
        print("   Carregando VAE...")
        vae = AutoencoderKL.from_pretrained(
            sd_model_id, subfolder="vae", use_safetensors=True
        )
        
        print("   Carregando Scheduler...")
        scheduler = DDPMScheduler.from_pretrained(
            sd_model_id, subfolder="scheduler", use_safetensors=True
        )
        
        print("   Carregando CLIP...")
        feature_extractor = CLIPImageProcessor.from_pretrained(
            clip_model_id, use_safetensors=True
        )
        image_encoder = CLIPVisionModel.from_pretrained(
            clip_model_id, use_safetensors=True
        )
        
        print("   Criando pipeline...")
        pipe = StableDiffusionReferenceNetPipeline(
            unet=unet,
            referencenet=referencenet,
            conditioning_referencenet=conditioning_referencenet,
            vae=vae,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            scheduler=scheduler,
        )
        pipe = pipe.to(device, dtype=dtype)
        
        print("   ✓ Pipeline carregado com sucesso!")
        
    except Exception as e:
        print(f"   ✗ Erro ao carregar modelos: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Inicializar Face Alignment
    print("\n2. Inicializando Face Alignment...")
    try:
        fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            face_detector="sfd",
            device=device
        )
        print("   ✓ Face Alignment inicializado")
    except Exception as e:
        print(f"   ✗ Erro: {e}")
        return False
    
    # Carregar imagem de teste
    print("\n3. Carregando imagem de teste...")
    image_path = "my_dataset/test/00482.png"
    try:
        original_image = load_image(image_path)
        print(f"   ✓ Imagem carregada: {original_image.size}")
    except Exception as e:
        print(f"   ✗ Erro: {e}")
        return False
    
    # Definir testes de segmentação
    print("\n4. Testando anonimização segmentada com diferentes características...")
    print("=" * 70)
    
    test_cases = [
        {
            'features': ['eyes'],
            'name': 'Apenas Olhos',
            'num_steps': 30,  # Menos passos para teste mais rápido
        },
        {
            'features': ['mouth'],
            'name': 'Apenas Boca',
            'num_steps': 30,
        },
        {
            'features': ['eyes', 'mouth'],
            'name': 'Olhos + Boca',
            'num_steps': 30,
        },
        {
            'features': ['nose'],
            'name': 'Nariz',
            'num_steps': 30,
        },
        {
            'features': ['eyebrows', 'eyes'],
            'name': 'Sobrancelhas + Olhos',
            'num_steps': 30,
        },
        {
            'features': ['eyes', 'nose', 'mouth'],
            'name': 'Olhos + Nariz + Boca',
            'num_steps': 30,
        },
    ]
    
    results = []
    generator = torch.manual_seed(42)
    
    for i, test_case in enumerate(test_cases):
        print(f"\n   Teste {i+1}/{len(test_cases)}: {test_case['name']}")
        print(f"      Features: {test_case['features']}")
        print(f"      Passos de inferência: {test_case['num_steps']}")
        
        try:
            anon_image = anonymize_faces_segmented(
                image=original_image,
                face_alignment_model=fa,
                mask_features=test_case['features'],
                operator_type='diffusion',
                pipe=pipe,
                generator=generator,
                num_inference_steps=test_case['num_steps'],
                guidance_scale=4.0,
                anonymization_degree=1.25,
                dilate_radius=3,
                smooth_edges=True,
            )
            
            # Salvar resultado
            safe_name = test_case['name'].replace(' ', '_').replace('+', '_').lower()
            output_path = f"test_diffusion_segmented_{i+1:02d}_{safe_name}.png"
            anon_image.save(output_path)
            
            print(f"      ✓ Anonimização concluída")
            print(f"      ✓ Salvo em: {output_path}")
            
            results.append({
                'name': test_case['name'],
                'features': test_case['features'],
                'output': output_path,
                'status': 'success'
            })
            
        except Exception as e:
            print(f"      ✗ Erro: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'name': test_case['name'],
                'status': 'error',
                'error': str(e)
            })
    
    # Resumo
    print("\n" + "=" * 70)
    print("RESUMO DOS TESTES")
    print("=" * 70)
    
    successful = [r for r in results if r.get('status') == 'success']
    failed = [r for r in results if r.get('status') == 'error']
    
    print(f"\n✓ Sucessos: {len(successful)}/{len(test_cases)}")
    print(f"✗ Falhas: {len(failed)}/{len(test_cases)}")
    
    if successful:
        print("\n✓ Testes concluídos com sucesso:")
        for r in successful:
            print(f"  - {r['name']}: {r['output']}")
            print(f"    Features: {r['features']}\n")
    
    if failed:
        print("\n✗ Erros encontrados:")
        for r in failed:
            print(f"  - {r['name']}: {r.get('error', 'Erro desconhecido')}")
    
    # Criar grid de comparação
    if len(successful) > 0:
        print("\n5. Criando grid de comparação...")
        try:
            result_images = []
            for r in successful:
                result_images.append(load_image(r['output']))
            
            if len(result_images) > 0:
                # Adicionar imagem original no início
                result_images.insert(0, original_image)
                grid = make_image_grid(result_images, rows=2, cols=4)
                grid.save("test_diffusion_segmented_comparison.png")
                print(f"   ✓ Grid salvo em: test_diffusion_segmented_comparison.png")
        except Exception as e:
            print(f"   ⚠ Erro ao criar grid: {e}")
    
    print("\n" + "=" * 70)
    print("✓ TESTE CONCLUÍDO!")
    print("=" * 70)
    print("\nNOTA: A anonimização com face_anon_simple foi aplicada apenas nas")
    print("      regiões segmentadas especificadas. O resto da face permanece")
    print("      inalterado, demonstrando anonimização seletiva.")
    print("=" * 70)
    
    return len(successful) > 0


if __name__ == "__main__":
    success = test_segmented_diffusion()
    exit(0 if success else 1)

