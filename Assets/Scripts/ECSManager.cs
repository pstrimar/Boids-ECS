using UnityEngine;
using Unity.Entities;
using Unity.Mathematics;
using Unity.Transforms;
using Unity.Rendering;
using Unity.Collections;

public class ECSManager : MonoBehaviour
{
    public static ECSManager Instance;

    public float cageLimits;
    public float boidSpeed;
    public float boidPerceptionRadius;
    public float separationWeight;
    public float cohesionWeight;
    public float alignmentWeight;
    public float avoidWallsWeight;
    public float avoidWallsTurnDist;

    [SerializeField]
    int numBoids = 200;

    [SerializeField]
    Mesh sharedMesh;

    [SerializeField]
    Material sharedMaterial;

    EntityManager entityManager;

    void Awake()
    {
        Instance = this;

        entityManager = World.DefaultGameObjectInjectionWorld.EntityManager;

        EntityArchetype boidArchetype = entityManager.CreateArchetype(
            typeof(BoidData),
            typeof(RenderMesh),
            typeof(RenderBounds),
            typeof(LocalToWorld)
            );

        NativeArray<Entity> boidArray = new NativeArray<Entity>(numBoids, Allocator.Temp);
        entityManager.CreateEntity(boidArchetype, boidArray);

        for (int i = 0; i < boidArray.Length; i++)
        {
            Unity.Mathematics.Random rand = new Unity.Mathematics.Random((uint)i + 1);

            entityManager.SetComponentData(boidArray[i], new LocalToWorld
            {
                Value = float4x4.TRS(
                    RandomPosition(),
                    RandomRotation(),
                    new float3(1f, 1f, 1f))
            });

            entityManager.SetSharedComponentData(boidArray[i], new RenderMesh
            {
                mesh = sharedMesh,
                material = sharedMaterial
            });            
        }
        boidArray.Dispose();
    }

    private float3 RandomPosition()
    {
        return new float3(
            UnityEngine.Random.Range(-cageLimits / 2f, cageLimits / 2f),
            UnityEngine.Random.Range(-cageLimits / 2f, cageLimits / 2f),
            UnityEngine.Random.Range(-cageLimits / 2f, cageLimits / 2f));
    }

    private quaternion RandomRotation()
    {
        return quaternion.Euler(
            UnityEngine.Random.Range(-360f, 360f),
            UnityEngine.Random.Range(-360f, 360f),
            UnityEngine.Random.Range(-360f, 360f));
    }

    private void OnDrawGizmos()
    {
        Gizmos.color = Color.green;
        Gizmos.DrawWireCube(Vector3.zero, new Vector3(cageLimits, cageLimits, cageLimits));
    }
}