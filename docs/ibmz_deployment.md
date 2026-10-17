# Confidential Federated Learning on IBM Z & LinuxONE

> **Challenge:** AI Secured: Innovation Without Exposure  
> **Datathon:** IBM Z Datathon 2026  
> **Status:** Architectural Design & Planned Integration *(Prototyped and verified on macOS/x86; not yet tested on physical IBM Z / LinuxONE hardware).*

---

## 1. Executive Summary & Challenge Context

Financial fraud detection across institutions faces a fundamental paradox:
- **The Data Silo Problem:** Fraud patterns evolve rapidly across multiple banks, merchant acquirers, and fintech rails. No individual institution observes sufficient fraudulent behavior in isolation to train a globally optimal detection model.
- **The Privacy & Regulatory Constraint:** Exchanging raw transaction records (containing PII, account numbers, cardholder details, and balances) violates strict privacy legislation, including GDPR, PCI-DSS Level 1, and CCPA, as well as banking secrecy laws.

The **federated-fraud-ibmz** project resolves this tension by marrying **Pure PyTorch Federated Learning (FedAvg)** with **Differential Privacy (DP)** and the confidential computing capabilities of **IBM Secure Execution for Linux** on IBM z16 and LinuxONE Emperor 4 / Rockhopper 4.

---

## 2. IBM Secure Execution Architecture

IBM Secure Execution for Linux is a hardware-based confidential computing technology that protects virtual machines (KVM guests) from unauthorized inspection or tampering by the host operating system, the KVM hypervisor, and system administrators.

```
+---------------------------------------------------------------------------------+
|                   IBM Z / LinuxONE Hardware Platform (s390x)                    |
|                                                                                 |
|  +---------------------------------------------------------------------------+  |
|  |                 Ultravisor (Hardware Firmware Security Level)             |  |
|  |   - Manages cryptographic keys bound to specific host hardware PU keys     |  |
|  |   - Enforces encrypted memory access control at the hardware page level   |  |
|  +---------------------------------------------------------------------------+  |
|                                                                                 |
|  +---------------------------------------------------------------------------+  |
|  |              CPACF (Central Processor Assist for Cryptographic Functions) |  |
|  |   - Hardware-accelerated AES-256 GCM encryption for memory & I/O          |  |
|  +---------------------------------------------------------------------------+  |
|                                                                                 |
|  +-------------------------------------+   +---------------------------------+  |
|  | Untrusted Host OS / KVM Hypervisor  |   | Untrusted Cloud Admin / Root    |  |
|  | (Cannot read or modify guest memory)|   | (Cannot dump memory or attach)  |  |
|  +-------------------------------------+   +---------------------------------+  |
|                   │                                       │                     |
|                   ▼ [Blocked by Ultravisor]               ▼ [Blocked by Hardware|
|  +───────────────────────────────────────────────────────────────────────────+  |
|  |                     SECURE EXECUTION ENCLAVE GUEST                        |  |
|  |  +---------------------------------------------------------------------+  |  |
|  |  | Linux Kernel + Initrd (Encrypted Boot Image built via genprotimg)   |  |  |
|  |  | Container Runtime: Docker (linux/s390x)                             |  |  |
|  |  | Application: Bank Node (Local PyTorch Training + DP Clip)            |  |  |
|  |  | Local Storage: Encrypted LUKS volume with proprietary bank data    |  |  |
|  |  +---------------------------------------------------------------------+  |  |
|  +───────────────────────────────────────────────────────────────────────────+  |
+---------------------------------------------------------------------------------+
```

### Key Security Guarantees:
1. **Memory Encryption in Flight and at Rest:** The Ultravisor hardware layer encrypts guest memory pages using hardware-managed AES keys. If a hypervisor admin dumps host physical RAM, the guest data appears as ciphertext.
2. **Attestation & Image Protection:** The guest image is prepared using `genprotimg`, which binds the kernel, ramdisk, and boot parameters using host-specific public keys. Only an authorized IBM Z machine can boot the image.
3. **Protected CPU Registers:** When the guest context-switches, CPU registers are automatically encrypted before the hypervisor gains control.
4. **Hardware Cryptographic Acceleration:** CPACF (Central Processor Assist for Cryptographic Functions) ensures near-zero latency overhead for cryptographic hashing (SHA-256) and tensor encryption.

---

## 3. End-to-End Federated Protocol Flow

In a multi-bank consortium (e.g., Bank 1 through Bank 4), each node runs as an independent Secure Execution guest.

```
       [Bank 1 Secure Guest]              [Central Aggregator Enclave]            [Bank 4 Secure Guest]
                │                                      │                                    │
   1. Local     │ (Local creditcard data               │                                    │
      Data      │  resides strictly in RAM)            │                                    │
                │                                      │                                    │
   2. Handshake ├─────── TLS 1.3 mTLS + Attestation ──►│◄── TLS 1.3 mTLS + Attestation ────┤
                │                                      │                                    │
   3. Distribute│◄────── Broadcast Global Weights W_0 ─┴────── Broadcast Global Weights ────►│
                │                                                                           │
   4. Local     │ [2 Epochs Local Training]                                                 │ [2 Epochs Local]
      Training  │ Delta_1 = W_local - W_0                                                   │ Delta_4 = W_local - W_0
                │ Clip L2 Norm: ||Delta_1|| <= C                                            │ Clip L2 Norm: ||Delta_4|| <= C
                │                                                                           │
   5. Update    ├─────── Transmit Clipped Delta_1 ────►│◄───── Transmit Clipped Delta_4 ────┤
                │                                      │                                    │
   6. Aggregate │                                      │ [Weighted FedAvg]                  │
      & DP      │                                      │ Delta_agg = sum(alpha_k * Delta_k) │
                │                                      │ Add Gaussian Noise N(0, sigma^2)   │
                │                                      │ W_1 = W_0 + Delta_agg              │
                │                                      │ Compute SHA-256 Hash of W_1        │
                │                                      │ Write to Immutable Audit Log       │
                │                                      │                                    │
   7. Next      │◄────── Broadcast Global Weights W_1 ─┴────── Broadcast Global Weights ────►│
      Round     │                                                                           │
```

---

## 4. Building and Deploying on IBM LinuxONE (`linux/s390x`)

### Step 1: Multi-Architecture Container Build
Using Docker Buildx to cross-compile the container image for the `linux/s390x` architecture:

```bash
docker buildx build \
  --platform linux/s390x \
  -t ibm-z-datathon/bank-node:latest \
  --file Dockerfile \
  --push my-secure-registry.internal/bank-node:latest
```

### Step 2: Preparing the IBM Secure Execution Image
On an authorized IBM LinuxONE management node, use the `genprotimg` tool from `s390-tools` to bind the container host OS to the target machine's host key document:

```bash
# Retrieve the host key document from the IBM Z machine
curl -o hostkey.crt https://ibmz-host.internal/keys/hkd-z16-1002.crt

# Build the protected boot image
genprotimg \
  -i /boot/vmlinuz-linux \
  -r /boot/initramfs-linux.img \
  -p "root=/dev/mapper/secure_luks quiet console=ttysclp0" \
  -k hostkey.crt \
  --cert-revocation-list ibm_crl.pem \
  -o /var/lib/libvirt/images/bank_secure_guest.img
```

### Step 3: Launching the Secure Execution Guest via Libvirt
Launch the guest domain with Secure Execution protection enabled (`<launchSecurity type="s390-pv"/>`):

```xml
<domain type='kvm'>
  <name>bank1-secure-node</name>
  <memory unit='GiB'>16</memory>
  <vcpu placement='static'>4</vcpu>
  <os>
    <type arch='s390x' machine='s390-ccw-virtio'>hvm</type>
    <kernel>/var/lib/libvirt/images/bank_secure_guest.img</kernel>
  </os>
  <launchSecurity type='s390-pv'/>
  ...
</domain>
```

---

## 5. Regulatory Compliance & Auditability

To satisfy strict financial compliance bodies (such as BaFin, OCC, EBA, and FINRA):
1. **Zero Data Movement:** Auditable proofs demonstrate that transactional data never travels across institutional boundaries.
2. **Cryptographic Model Hash (`SHA-256`):** After each round, the aggregator calculates a deterministic digest of the weights:
   ```json
   {
     "timestamp": "2026-10-17T13:00:00Z",
     "round": 15,
     "participating_banks": ["Bank_1", "Bank_2", "Bank_3", "Bank_4"],
     "update_norm_per_bank": {"Bank_1": 0.4281, "Bank_2": 0.3912, "Bank_3": 0.5120, "Bank_4": 0.4873},
     "noise_multiplier": 0.1,
     "global_model_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
     "test_auprc": 0.8421
   }
   ```
   This ledger is persisted in `results/audit_log.jsonl` and can be mirrored to a permissioned blockchain (e.g., Hyperledger Fabric running on IBM Z).

---

## 6. Implementation Status & Limitations

> [!IMPORTANT]
> **Planned Integration Disclaimer**:
> This project provides the complete algorithmic foundation, container definitions, and architectural specification for IBM Secure Execution on IBM Z / LinuxONE.
> Execution has been prototyped and rigorously benchmarked on macOS (Apple Silicon CPU) under PyTorch.
> Hardware-specific Secure Execution validation (invoking `genprotimg` on an actual IBM z16 LPAR) represents planned production integration for the Datathon deployment phase.
