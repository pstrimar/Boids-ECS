using Unity.Collections;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Transforms;

public class BoidMovementSystem : JobComponentSystem
{
    private ECSManager controller;

    private struct EntityWithLocalToWorld
    {
        public Entity entity;
        public LocalToWorld localToWorld;
    }

    protected override JobHandle OnUpdate(JobHandle inputDeps)
    {
        if (!controller)
            controller = ECSManager.Instance;

        if (controller)
        {
            EntityQuery boidQuery = GetEntityQuery(ComponentType.ReadOnly<BoidData>(), ComponentType.ReadOnly<LocalToWorld>());

            NativeArray<Entity> entityArray = boidQuery.ToEntityArray(Allocator.TempJob);
            NativeArray<LocalToWorld> localToWorldArray = boidQuery.ToComponentDataArray<LocalToWorld>(Allocator.TempJob);

            NativeArray<EntityWithLocalToWorld> boidArray = new NativeArray<EntityWithLocalToWorld>(entityArray.Length, Allocator.TempJob);
            NativeArray<float4x4> newBoidTransforms = new NativeArray<float4x4>(entityArray.Length, Allocator.TempJob);

            for (int i = 0; i < entityArray.Length; i++)
            {
                boidArray[i] = new EntityWithLocalToWorld
                {
                    entity = entityArray[i],
                    localToWorld = localToWorldArray[i]
                };
            }

            entityArray.Dispose();
            localToWorldArray.Dispose();

            float cageLimits = controller.cageLimits;
            float boidSpeed = controller.boidSpeed;
            float boidPerceptionRadius = controller.boidPerceptionRadius;
            float separationWeight = controller.separationWeight;
            float cohesionWeight = controller.cohesionWeight;
            float alignmentWeight = controller.alignmentWeight;
            float avoidWallsWeight = controller.avoidWallsWeight;
            float avoidWallsTurnDist = controller.avoidWallsTurnDist;

            float deltaTime = Time.DeltaTime;

            JobHandle boidJobHandle = Entities
                .WithAll<BoidData>()
                .WithBurst()
                .WithReadOnly(boidArray)
                .WithDisposeOnCompletion(boidArray)
                .ForEach((Entity boid, int entityInQueryIndex, ref LocalToWorld localToWorld) =>
                {
                    float3 boidPosition = localToWorld.Position;
                    float3 separationSum = float3.zero;
                    float3 positionSum = float3.zero;
                    float3 headingSum = float3.zero;

                    int boidsNearby = 0;

                    for (int otherBoidIndex = 0; otherBoidIndex < boidArray.Length; otherBoidIndex++)
                    {
                        if (boid != boidArray[otherBoidIndex].entity)
                        {
                            float3 otherPosition = boidArray[otherBoidIndex].localToWorld.Position;
                            float distToOtherBoid = math.length(boidPosition - otherPosition);

                            if (distToOtherBoid < boidPerceptionRadius)
                            {
                                // avoid dividing by zero
                                separationSum += (boidPosition - otherPosition) * (1f / math.max(distToOtherBoid, .0001f));
                                positionSum += otherPosition;
                                headingSum += boidArray[otherBoidIndex].localToWorld.Forward;

                                boidsNearby++;
                            }
                        }
                    }

                    float3 force = float3.zero;

                    if (boidsNearby > 0)
                    {
                        // move away from average of neighbors position
                        force += (separationSum / boidsNearby) * separationWeight;
                        // move towards center of group
                        force += ((positionSum / boidsNearby) - boidPosition) * cohesionWeight;
                        // move towards average heading of group
                        force += (headingSum / boidsNearby) * alignmentWeight;
                    }
                    if (math.min(math.min(
                        (cageLimits / 2f) - math.abs(boidPosition.x),
                        (cageLimits / 2f) - math.abs(boidPosition.y)),
                        (cageLimits / 2f) - math.abs(boidPosition.z))
                        < avoidWallsTurnDist)
                    {
                        // move towards center of bounds if too close to the limit
                        force += -math.normalize(boidPosition) * avoidWallsWeight;
                    }

                    float3 velocity = localToWorld.Forward * boidSpeed;
                    velocity += force * deltaTime;
                    velocity = math.normalize(velocity) * boidSpeed;

                    newBoidTransforms[entityInQueryIndex] = float4x4.TRS(
                        localToWorld.Position + velocity * deltaTime,
                        quaternion.LookRotationSafe(velocity, localToWorld.Up),
                        new float3(1f, 1f, 1f)
                        );
                }).Schedule(inputDeps);

            JobHandle boidMoveJob = Entities
                .WithAll<BoidData>()
                .WithBurst()
                .WithReadOnly(newBoidTransforms)
                .WithDisposeOnCompletion(newBoidTransforms)
                .ForEach((Entity boid, int entityInQueryIndex, ref LocalToWorld localToWorld) =>
                {
                    localToWorld.Value = newBoidTransforms[entityInQueryIndex];
                }).Schedule(boidJobHandle);

            return boidMoveJob;
        }
        else
            return inputDeps;
    }
}
