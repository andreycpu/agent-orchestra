"""
Cluster management for distributed Agent Orchestra deployments
"""
import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import structlog

from .types import AgentInfo, Task, TaskStatus
from .events import EventBus, Event

logger = structlog.get_logger(__name__)


class NodeStatus(str, Enum):
    """Status of cluster nodes"""
    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"


class ClusterRole(str, Enum):
    """Roles that nodes can have in the cluster"""
    COORDINATOR = "coordinator"
    WORKER = "worker"
    EDGE = "edge"
    BACKUP = "backup"


@dataclass
class ClusterNode:
    """Information about a cluster node"""
    id: str
    hostname: str
    ip_address: str
    port: int
    role: ClusterRole
    status: NodeStatus
    capabilities: List[str]
    agent_count: int
    load: float  # 0.0 to 1.0
    last_heartbeat: datetime
    metadata: Dict[str, Any]
    version: str = "0.1.0"
    
    def is_healthy(self, heartbeat_timeout_seconds: int = 60) -> bool:
        """Check if node is healthy based on heartbeat"""
        if self.status == NodeStatus.OFFLINE:
            return False
        
        time_since_heartbeat = (datetime.utcnow() - self.last_heartbeat).total_seconds()
        return time_since_heartbeat < heartbeat_timeout_seconds


@dataclass
class ClusterStats:
    """Overall cluster statistics"""
    total_nodes: int
    healthy_nodes: int
    total_agents: int
    total_tasks_pending: int
    total_tasks_running: int
    average_load: float
    cluster_health_score: float


class LeaderElection:
    """Leader election for cluster coordination"""
    
    def __init__(self, node_id: str, redis_client=None):
        self.node_id = node_id
        self.redis = redis_client
        self._is_leader = False
        self._election_key = "cluster:leader"
        self._heartbeat_key = f"cluster:leader:heartbeat"
        self._lease_duration = 30  # seconds
        
    async def start_election(self):
        """Start leader election process"""
        if not self.redis:
            # Without Redis, assume single node is leader
            self._is_leader = True
            return True
        
        try:
            # Try to acquire leadership
            result = await self.redis.set(
                self._election_key,
                self.node_id,
                nx=True,  # Only set if doesn't exist
                ex=self._lease_duration
            )
            
            if result:
                self._is_leader = True
                logger.info("Elected as cluster leader", node_id=self.node_id)
                
                # Start heartbeat to maintain leadership
                asyncio.create_task(self._leader_heartbeat_loop())
                return True
            else:
                self._is_leader = False
                current_leader = await self.redis.get(self._election_key)
                logger.info("Another node is leader", 
                           node_id=self.node_id,
                           current_leader=current_leader.decode() if current_leader else "unknown")
                return False
                
        except Exception as e:
            logger.error("Leader election failed", node_id=self.node_id, error=str(e))
            return False
    
    async def _leader_heartbeat_loop(self):
        """Maintain leadership through heartbeats"""
        while self._is_leader:
            try:
                # Extend lease
                await self.redis.expire(self._election_key, self._lease_duration)
                await self.redis.set(self._heartbeat_key, time.time(), ex=self._lease_duration)
                
                await asyncio.sleep(self._lease_duration // 2)  # Heartbeat at half lease duration
                
            except Exception as e:
                logger.error("Leader heartbeat failed", error=str(e))
                self._is_leader = False
                break
    
    def is_leader(self) -> bool:
        """Check if this node is the current leader"""
        return self._is_leader
    
    async def get_current_leader(self) -> Optional[str]:
        """Get current cluster leader"""
        if not self.redis:
            return self.node_id if self._is_leader else None
        
        try:
            leader = await self.redis.get(self._election_key)
            return leader.decode() if leader else None
        except:
            return None


class ClusterManager:
    """Manages cluster membership and coordination"""
    
    def __init__(self, node_id: str, hostname: str, ip_address: str, port: int,
                 role: ClusterRole = ClusterRole.WORKER, redis_client=None, event_bus=None):
        self.node_id = node_id
        self.hostname = hostname
        self.ip_address = ip_address
        self.port = port
        self.role = role
        self.redis = redis_client
        self.event_bus = event_bus or EventBus()
        
        self._nodes: Dict[str, ClusterNode] = {}
        self._local_node: ClusterNode = None
        self._leader_election = LeaderElection(node_id, redis_client)
        self._running = False
        
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._discovery_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        self._capabilities: Set[str] = set()
        self._agent_count = 0
        self._load = 0.0
    
    async def start(self):
        """Start cluster management"""
        self._running = True
        
        # Create local node info
        self._local_node = ClusterNode(
            id=self.node_id,
            hostname=self.hostname,
            ip_address=self.ip_address,
            port=self.port,
            role=self.role,
            status=NodeStatus.ONLINE,
            capabilities=list(self._capabilities),
            agent_count=self._agent_count,
            load=self._load,
            last_heartbeat=datetime.utcnow(),
            metadata={}
        )
        
        # Register local node
        await self._register_node(self._local_node)
        
        # Start background tasks
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._discovery_task = asyncio.create_task(self._node_discovery_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        # Participate in leader election
        await self._leader_election.start_election()
        
        logger.info("Cluster manager started",
                   node_id=self.node_id,
                   role=self.role,
                   is_leader=self._leader_election.is_leader())
    
    async def stop(self):
        """Stop cluster management"""
        self._running = False
        
        # Mark node as offline
        if self._local_node:
            self._local_node.status = NodeStatus.OFFLINE
            await self._register_node(self._local_node)
        
        # Cancel background tasks
        for task in [self._heartbeat_task, self._discovery_task, self._cleanup_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("Cluster manager stopped", node_id=self.node_id)
    
    async def _register_node(self, node: ClusterNode):
        """Register/update a node in the cluster"""
        self._nodes[node.id] = node
        
        if self.redis:
            try:
                # Store node info in Redis
                node_data = asdict(node)
                node_data["last_heartbeat"] = node.last_heartbeat.isoformat()
                
                await self.redis.hset(
                    "cluster:nodes",
                    node.id,
                    json.dumps(node_data, default=str)
                )
                
                # Set TTL for node entry
                await self.redis.expire(f"cluster:node:{node.id}", 120)  # 2 minutes
                
            except Exception as e:
                logger.error("Failed to register node in Redis",
                           node_id=node.id, error=str(e))
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats"""
        while self._running:
            try:
                # Update local node status
                self._local_node.last_heartbeat = datetime.utcnow()
                self._local_node.agent_count = self._agent_count
                self._local_node.load = self._load
                self._local_node.capabilities = list(self._capabilities)
                
                # Register updated info
                await self._register_node(self._local_node)
                
                # Emit heartbeat event
                await self.event_bus.emit(
                    "cluster.node_heartbeat",
                    self.node_id,
                    {"node": asdict(self._local_node)}
                )
                
                await asyncio.sleep(30)  # Heartbeat every 30 seconds
                
            except Exception as e:
                logger.error("Heartbeat failed", error=str(e))
                await asyncio.sleep(5)  # Shorter retry interval on error
    
    async def _node_discovery_loop(self):
        """Discover other nodes in the cluster"""
        while self._running:
            try:
                if self.redis:
                    # Get all nodes from Redis
                    nodes_data = await self.redis.hgetall("cluster:nodes")
                    
                    discovered_nodes = {}
                    for node_id, node_json in nodes_data.items():
                        try:
                            node_data = json.loads(node_json.decode())
                            node_data["last_heartbeat"] = datetime.fromisoformat(
                                node_data["last_heartbeat"]
                            )
                            
                            node = ClusterNode(**node_data)
                            discovered_nodes[node_id.decode()] = node
                            
                        except Exception as e:
                            logger.warning("Failed to parse node data",
                                         node_id=node_id.decode(), error=str(e))
                    
                    # Update nodes and check for changes
                    old_nodes = set(self._nodes.keys())
                    new_nodes = set(discovered_nodes.keys())
                    
                    # New nodes joined
                    for node_id in new_nodes - old_nodes:
                        if node_id != self.node_id:  # Don't emit for self
                            await self.event_bus.emit(
                                "cluster.node_joined",
                                self.node_id,
                                {"node": asdict(discovered_nodes[node_id])}
                            )
                    
                    # Nodes left
                    for node_id in old_nodes - new_nodes:
                        if node_id != self.node_id:  # Don't emit for self
                            await self.event_bus.emit(
                                "cluster.node_left",
                                self.node_id,
                                {"node_id": node_id}
                            )
                    
                    self._nodes = discovered_nodes
                
                await asyncio.sleep(60)  # Discovery every minute
                
            except Exception as e:
                logger.error("Node discovery failed", error=str(e))
                await asyncio.sleep(30)
    
    async def _cleanup_loop(self):
        """Clean up dead nodes"""
        while self._running:
            try:
                current_time = datetime.utcnow()
                dead_nodes = []
                
                for node_id, node in self._nodes.items():
                    if node_id != self.node_id and not node.is_healthy():
                        dead_nodes.append(node_id)
                
                # Remove dead nodes
                for node_id in dead_nodes:
                    del self._nodes[node_id]
                    
                    if self.redis:
                        await self.redis.hdel("cluster:nodes", node_id)
                    
                    await self.event_bus.emit(
                        "cluster.node_removed",
                        self.node_id,
                        {"node_id": node_id, "reason": "heartbeat_timeout"}
                    )
                    
                    logger.info("Removed dead node", node_id=node_id)
                
                await asyncio.sleep(120)  # Cleanup every 2 minutes
                
            except Exception as e:
                logger.error("Cleanup loop failed", error=str(e))
                await asyncio.sleep(60)
    
    def get_cluster_stats(self) -> ClusterStats:
        """Get cluster statistics"""
        total_nodes = len(self._nodes)
        healthy_nodes = sum(1 for node in self._nodes.values() if node.is_healthy())
        total_agents = sum(node.agent_count for node in self._nodes.values())
        
        if healthy_nodes > 0:
            average_load = sum(node.load for node in self._nodes.values() 
                             if node.is_healthy()) / healthy_nodes
        else:
            average_load = 0.0
        
        # Calculate cluster health score (0.0 to 1.0)
        if total_nodes > 0:
            health_score = healthy_nodes / total_nodes
            # Adjust for load
            health_score *= (1.0 - min(average_load, 1.0) * 0.5)
        else:
            health_score = 0.0
        
        return ClusterStats(
            total_nodes=total_nodes,
            healthy_nodes=healthy_nodes,
            total_agents=total_agents,
            total_tasks_pending=0,  # Would be populated from task queue
            total_tasks_running=0,  # Would be populated from running tasks
            average_load=average_load,
            cluster_health_score=health_score
        )
    
    def get_nodes(self, role: Optional[ClusterRole] = None, 
                  status: Optional[NodeStatus] = None) -> List[ClusterNode]:
        """Get nodes filtered by role and/or status"""
        nodes = list(self._nodes.values())
        
        if role:
            nodes = [node for node in nodes if node.role == role]
        
        if status:
            nodes = [node for node in nodes if node.status == status]
        
        return nodes
    
    def get_node(self, node_id: str) -> Optional[ClusterNode]:
        """Get specific node by ID"""
        return self._nodes.get(node_id)
    
    def is_leader(self) -> bool:
        """Check if this node is the cluster leader"""
        return self._leader_election.is_leader()
    
    async def get_leader(self) -> Optional[str]:
        """Get current cluster leader"""
        return await self._leader_election.get_current_leader()
    
    def update_capabilities(self, capabilities: Set[str]):
        """Update node capabilities"""
        self._capabilities = capabilities
    
    def update_agent_count(self, count: int):
        """Update agent count on this node"""
        self._agent_count = count
    
    def update_load(self, load: float):
        """Update node load (0.0 to 1.0)"""
        self._load = max(0.0, min(1.0, load))
    
    async def find_best_node_for_task(self, task: Task) -> Optional[str]:
        """Find the best node to execute a task"""
        if not self.is_leader():
            return None  # Only leader should route tasks
        
        suitable_nodes = []
        
        for node in self._nodes.values():
            if (node.is_healthy() and 
                node.status == NodeStatus.ONLINE and
                node.load < 0.9):  # Don't assign to overloaded nodes
                
                # Check if node has required capabilities
                # (This would need to be enhanced based on task requirements)
                suitable_nodes.append((node, node.load))
        
        if not suitable_nodes:
            return None
        
        # Sort by load (prefer less loaded nodes)
        suitable_nodes.sort(key=lambda x: x[1])
        
        return suitable_nodes[0][0].id
    
    async def set_maintenance_mode(self, enable: bool, reason: str = ""):
        """Set node maintenance mode"""
        if enable:
            self._local_node.status = NodeStatus.MAINTENANCE
            self._local_node.metadata["maintenance_reason"] = reason
            self._local_node.metadata["maintenance_started"] = datetime.utcnow().isoformat()
        else:
            self._local_node.status = NodeStatus.ONLINE
            self._local_node.metadata.pop("maintenance_reason", None)
            self._local_node.metadata.pop("maintenance_started", None)
        
        await self._register_node(self._local_node)
        
        await self.event_bus.emit(
            "cluster.node_maintenance",
            self.node_id,
            {"enabled": enable, "reason": reason}
        )
        
        logger.info("Maintenance mode changed",
                   node_id=self.node_id,
                   enabled=enable,
                   reason=reason)
    
    def get_cluster_topology(self) -> Dict[str, Any]:
        """Get cluster topology information"""
        coordinators = self.get_nodes(role=ClusterRole.COORDINATOR)
        workers = self.get_nodes(role=ClusterRole.WORKER)
        edges = self.get_nodes(role=ClusterRole.EDGE)
        
        return {
            "total_nodes": len(self._nodes),
            "coordinators": [asdict(node) for node in coordinators],
            "workers": [asdict(node) for node in workers],
            "edges": [asdict(node) for node in edges],
            "leader": await self.get_leader(),
            "local_node": asdict(self._local_node) if self._local_node else None
        }


class TaskDistributor:
    """Distributes tasks across cluster nodes"""
    
    def __init__(self, cluster_manager: ClusterManager, redis_client=None):
        self.cluster_manager = cluster_manager
        self.redis = redis_client
        self._distribution_strategy = "load_balanced"  # round_robin, load_balanced, capability_aware
    
    async def distribute_task(self, task: Task) -> Optional[str]:
        """Distribute a task to the best available node"""
        if not self.cluster_manager.is_leader():
            return None  # Only leader distributes tasks
        
        target_node = await self.cluster_manager.find_best_node_for_task(task)
        
        if target_node:
            # Send task to target node via Redis
            if self.redis:
                task_data = {
                    "task": asdict(task),
                    "target_node": target_node,
                    "distributed_at": time.time()
                }
                
                await self.redis.lpush(
                    f"cluster:tasks:{target_node}",
                    json.dumps(task_data, default=str)
                )
                
                logger.info("Task distributed",
                           task_id=task.id,
                           target_node=target_node)
        
        return target_node