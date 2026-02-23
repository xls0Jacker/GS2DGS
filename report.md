_<font style="color:#000000;">[</font>__<font style="color:rgb(0, 0, 0);">Accepted by CVPR2025</font>__<font style="color:#000000;">]</font>_

<font style="color:#000000;">论文链接</font><font style="color:rgb(0, 0, 0);">：</font>[https://arxiv.org/abs/2506.13110](https://arxiv.org/abs/2506.13110)<font style="color:rgb(54, 54, 54);">  
</font><font style="color:#000000;">项目链接：</font>[https://github.com/hirotong/GS2DGS](https://github.com/hirotong/GS2DGS)

# 论文内容概述
**摘要**——高反射物体的三维建模由于其强烈的_视角相关外观 _而一直具有挑战性。尽管<u>以 </u>_<u>SDF</u>_<u> 为基础的方法能够重建出高质量的网格模型，但通常计算开销较大，且容易产生过度平滑的表面</u>。相比之下，<u>3D Gaussian Splatting（3DGS）具有速度快和高质量实时渲染的优势，但由于缺乏几何约束，从高斯表示中提取表面时往往会产生较大的噪声</u>。为弥合上述方法之间的差距，我们提出了一种基于 2D Gaussian Splatting（2DGS）的新型反射物体重建方法——GS-2DGS。该方法在保留 Gaussian Splatting 快速渲染能力的同时，引入来自基础模型的额外几何信息进行约束。合成数据和真实数据集上的实验结果表明，我们的方法在重建质量和重光照效果方面显著优于现有基于高斯的技术，同时在计算速度上比基于 SDF 的方法快一个数量级，但性能可与其相媲美。

<details class="lake-collapse"><summary id="u688bdfc6"><em><span class="ne-text">视角相关外观 （view-dependent appearance）  </span></em><span class="ne-text">：高反射物体（如金属）从不同角度观察，其</span><span class="ne-text" style="text-decoration: underline">颜色、亮度和纹理</span><span class="ne-text">均会发生变化。</span></summary><p id="u13f57bb3" class="ne-p"><img src="https://cdn.nlark.com/yuque/0/2026/png/45861457/1771659649730-9155b852-f0de-4c06-8903-202b9dee79fa.png" width="831.7310657747712" title="" crop="0,0,1,1" id="u8ddd04c8" class="ne-image"></p></details>
<details class="lake-collapse"><summary id="u6f0f13c6"><em><span class="ne-text">SDF（Signed Distance Function），有符号距离函数 </span></em><span class="ne-text">：输入三维空间中任意一点坐标，输出该坐标到物体表面的符号距离，在物体外部为正数，在物体内部为负数，为 0 时恰好在物体表面。</span></summary><p id="udb2f9e27" class="ne-p"><img src="https://cdn.nlark.com/yuque/0/2026/png/45861457/1771659834072-db9e9ea5-434a-47eb-974a-08f0c0bd0bd3.png" width="793.8151006036614" title="" crop="0,0,1,1" id="ZuMtL" class="ne-image"></p></details>
## Introduction
<u>从多视角图像中构建高反射物体的三维模型一直是计算机图形学与计算机视觉领域中的一项长期且具有挑战性的任务，这是因为镜面反射具有强烈的视角相关性，从而违背了大多数重建方法所采用的多视角一致性假设</u>。已有工作 _[28, 32]_ 通过引入神经辐射场（NeRF）和有符号距离场（SDF）取得了较为理想的效果。然而，这类方法在训练过程中通常需要消耗大量的计算资源，单个场景往往需要数小时才能完成训练。**<u>反射物体重建的困难根源在于该问题本身具有病态性：反射表面的外观由表面属性（材料与几何形状）以及环境光照条件共同决定</u>**<u>，而已有方法 </u>_[28, 32]_<u> 通常只考虑了其中的一个因素</u>。

<details class="lake-collapse"><summary id="u8ffa2089"><em><span class="ne-text">[28, 32]</span></em></summary><p id="ufef9d39c" class="ne-p"><span class="ne-text">[28] </span><strong><span class="ne-text">TensoSDF: Roughness-aware Tensorial Representation for Robust Geometry and Material Reconstruction</span></strong><span class="ne-text">（ACM Transactions on Graphics, 2024）：提出一种粗糙度感知的张量化 SDF 表示方法，将几何与材质属性（尤其是表面粗糙度）进行统一建模，通过张量分解实现高效表达与优化，在强反射与复杂光照条件下显著提升几何重建与材质恢复的鲁棒性与一致性，为可微分渲染框架下的联合重建提供了更稳定的表示形式。</span></p><p id="u02efb016" class="ne-p"><span class="ne-text">[32] </span><strong><span class="ne-text">NeRO: Neural Geometry and BRDF Reconstruction of Reflective Objects from Multiview Images</span></strong><span class="ne-text">（ACM Transactions on Graphics, 2023）：提出 NeRO 框架，在多视图图像输入下联合优化隐式几何表示与可学习 BRDF 模型，实现对高反射物体的几何与材质精确重建；通过物理一致的可微渲染约束与神经反射模型设计，有效缓解镜面高光带来的歧义问题，在复杂反射场景中显著提升重建精度与外观一致性。</span></p></details>
由于 <u>3D Gaussian Splatting（3DGS）在渲染速度、细节表现和照片一致性方面具有显著优势</u>，其已被广泛应用于三维场景建模。<u>然而，3DGS 的表面重建质量仍有待进一步提升</u>。<u>一方面，一些基于高斯的方法</u> _[10, 20, 50]_ <u>通过对高斯进行扁平化处理或引入 SDF 以增强表面平滑性，</u>从而在表面几何上施加额外约束，<u>以提升重建质量</u>。<u>这类方法在常见多视角图像上能够在大幅提高计算效率的同时获得更优的重建效果</u>。<u>另一方面</u>，另<u>一些工作 </u>_[18, 22, 31]_<u> 则通过引入基于物理的渲染（</u>_<u>PBR</u>_<u>）和逆渲染（IR）技术，提升了基于高斯方法的渲染性能并实现了重光照</u>。然而需要指出的是，**这些方法在反射物体重建方面仍然存在局限性，因为它们通常只关注问题的单一方面**。

<details class="lake-collapse"><summary id="u44d2f4b9"><em><span class="ne-text">扁平化处理（flattening the Gaussians）</span></em><span class="ne-text">：原始高斯在三个方向都有较大方差，形状类似椭球体；扁平化就是在法线方向上把方差压得非常小，只保留切平面方向的扩展。最后得到一个很薄的一层的“二维高斯盘“。（仍是三维，不等于 2DGS）</span></summary><p id="u328d3f28" class="ne-p"><img src="https://cdn.nlark.com/yuque/0/2026/png/45861457/1771661304326-503b1379-9d3b-4616-abe4-764b7ffbff1c.png" width="809.1428312047484" title="" crop="0,0,1,1" id="u89c231f9" class="ne-image"></p></details>
<details class="lake-collapse"><summary id="ud9b1e84d"><em><span class="ne-text">PBR (Physically Based Rendering)，基于物理的渲染 </span></em><span class="ne-text">；</span><em><span class="ne-text">IR (Inverse Rendering)，逆渲染 </span></em><span class="ne-text">：PBR 是基于几何、材质、光照等计算得到图像，IR 是通过图像计算几何、材质、光照等性质。通常 PBR 和 IR 一起使用，先 IR 得到属性，再通过 PBR 进行监督学习。</span></summary><p id="u03573db6" class="ne-p"><img src="https://cdn.nlark.com/yuque/0/2026/png/45861457/1771661544102-fcdf26b3-ad45-47d6-9506-cc7bb20d90d3.png" width="818.0167805001145" title="" crop="0,0,1,1" id="u10278629" class="ne-image"></p></details>
<details class="lake-collapse"><summary id="u90e30d63"><em><span class="ne-text">[10, 20, 50]</span></em></summary><p id="u934ccc6c" class="ne-p"><span class="ne-text">[10] </span><strong><span class="ne-text">High-Quality Surface Reconstruction Using Gaussian Surfels</span></strong><span class="ne-text">（ACM SIGGRAPH 2024 Conference Papers（SIGGRAPH）, 2024）：提出基于 Gaussian surfels 的高质量表面重建方法，将高斯原语与表面片元表示相结合，在保持高效渲染能力的同时强化几何一致性与法线约束，实现更平滑且细节丰富的网格重建效果，显著缓解传统 3DGS 表面提取噪声问题。</span></p><p id="u66708757" class="ne-p"><span class="ne-text">[20] </span><strong><span class="ne-text">2D Gaussian Splatting for Geometrically Accurate Radiance Fields</span></strong><span class="ne-text">（ACM SIGGRAPH 2024 Conference Papers（SIGGRAPH）, 2024）：提出 2D Gaussian Splatting（2DGS），将三维体高斯表示转化为二维平面高斯圆盘以增强几何表达能力，通过显式表面结构约束提升辐射场的几何精度，在保持实时渲染效率的同时显著改善表面重建质量，为几何与渲染统一优化提供了新范式。</span></p><p id="u668df356" class="ne-p"><span class="ne-text">[50] </span><strong><span class="ne-text">GSDF: 3DGS Meets SDF for Improved Rendering and Reconstruction</span></strong><span class="ne-text">（arXiv, 2024）：提出将 3D Gaussian Splatting 与 Signed Distance Function（SDF）相结合的混合表示框架，在保持 3DGS 高效渲染优势的同时引入 SDF 的显式几何约束，通过联合优化实现更精确的表面重建与更稳定的体渲染结果，在几何一致性与视觉质量之间取得更优平衡。</span></p></details>
<details class="lake-collapse"><summary id="u1bdd7896"><em><span class="ne-text">[18, 22, 31]</span></em></summary><p id="u560ea552" class="ne-p"><span class="ne-text">[18] </span><strong><span class="ne-text">Relightable 3D Gaussian: Real-time Point Cloud Relighting with BRDF Decomposition and Ray Tracing</span></strong><span class="ne-text">（arXiv, 2023）：提出可重光照的高斯点云重照明框架，通过 BRDF 分解结合光线追踪渲染实现实时点云重光照效果，使 3D 高斯表示能够显式建模材质反射特性，在动态光照场景下保持较高的渲染真实性与计算效率。</span></p><p id="u99d1d4c6" class="ne-p"><span class="ne-text">[22] </span><strong><span class="ne-text">GaussianShader: 3D Gaussian Splatting with Shading Functions for Reflective Surfaces</span></strong><span class="ne-text">（IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024）：提出 GaussianShader 方法，在 3D Gaussian Splatting 中引入可学习的着色函数以增强对高反射表面的建模能力，通过显式光照响应函数替代传统球谐函数，从而改善复杂反射材料下的渲染质量与视觉一致性。</span></p><p id="ucf975142" class="ne-p"><span class="ne-text">[31] </span><strong><span class="ne-text">GS-IR: 3D Gaussian Splatting for Inverse Rendering</span></strong><span class="ne-text">（IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024）：提出用于逆向渲染的 3D Gaussian Splatting 框架，通过联合优化几何、材质与光照参数实现物理一致的场景重建，在多视图条件下有效提升反射场景的重建精度与光照解耦能力。</span></p></details>
<u>由于反射物体重建问题本身的</u>_<u>病态性</u>_<u>，简单地将上述两个方面直接结合并不足以有效解决该问题。为此，一种更为直接的思路是对表面属性或光照条件引入额外约束</u>。近年来，<u>基于 Transformer 或扩散模型的方法</u> _[3, 16, 25, 34]_ 利用大量真实与合成数据进行训练，在单目几何（深度与法线）估计任务中展现出了令人瞩目的性能。<u>与依赖多视角一致性推断几何信息的多视角立体方法不同，这类方法基于大规模训练数据所蕴含的先验知识，能够直接从单张图像中预测几何信息，从而对反射表面问题具有更强的鲁棒性</u><font style="color:#8CCF17;">（说明为什么需要引入基础模型）</font>。

<details class="lake-collapse"><summary id="ufc87bcd8"><em><span class="ne-text">病态性</span></em><span class="ne-text">：反射表面的外观由表面属性（材料与几何形状）以及环境光照条件共同决定。</span></summary><p id="uaf1815c8" class="ne-p"><br></p></details>
<details class="lake-collapse"><summary id="u5da5a68c"><em><span class="ne-text">[3, 16, 25, 34]</span></em></summary><p id="u780f1d90" class="ne-p"><span class="ne-text">[3] </span><strong><span class="ne-text">Depth Pro: Sharp Monocular Metric Depth in Less Than a Second</span></strong><span class="ne-text">（arXiv, 2024）：提出高精度快速单目深度估计模型，通过优化网络结构与推理策略实现亚秒级深度预测，同时强化边缘细节恢复能力，使深度图具有更高的几何锐度与度量一致性，在实时场景深度感知中表现出较强实用价值。</span></p><p id="u789727b8" class="ne-p"><span class="ne-text">[16] </span><strong><span class="ne-text">GeoWizard: Unleashing the Diffusion Priors for 3D Geometry Estimation from a Single Image</span></strong><span class="ne-text">（European Conference on Computer Vision, 2025）：首次系统性地将扩散模型先验引入单图三维几何估计任务，利用生成式扩散特征增强几何结构恢复能力，在弱纹理与遮挡区域显著提升单目三维重建的鲁棒性与完整性。</span></p><p id="u1ea62d42" class="ne-p"><span class="ne-text">[25] </span><strong><span class="ne-text">Repurposing Diffusion-based Image Generators for Monocular Depth Estimation</span></strong><span class="ne-text">（IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024）：探索将预训练扩散图像生成模型重新用于单目深度估计，通过特征迁移与条件优化机制利用生成模型中隐含的结构先验，提高深度预测的全局一致性与精度。</span></p><p id="u0f495c0a" class="ne-p"><span class="ne-text">[34] </span><strong><span class="ne-text">Fine-tuning Image-conditional Diffusion Models is Easier than You Think</span></strong><span class="ne-text">（IEEE/CVF Winter Conference on Applications of Computer Vision, 2025）：提出简化的图像条件扩散模型微调策略，降低扩散模型在下游视觉任务中的训练复杂度，在保持生成质量的同时提升模型适配效率，推动扩散模型在实际视觉应用中的落地。</span></p></details>
**本文提出了一种新的反射物体重建框架，通过在基于物理的高斯点绘制中引入基础模型（foundation models）来实现。通过融合额外的表面信息，我们的方法能够同时应对由表面相关属性（几何与材料）以及光照条件所带来的问题**。实验结果表明，与现有基于 Gaussian Splatting 的方法相比，我们的方法在性能上具有明显优势；同时，在保持计算效率提升一个数量级的前提下，其重建效果可与基于 NeRF 的方法相当。

**综上，本文的主要贡献如下**：  
• <u>提出了一种利用基础模型的、</u>_<u>基于物理的</u>_<u> Gaussian Splatting 的反射物体重建新框架</u>。  
• <u>引入</u>_<u>延迟着色（deferred shading）</u>_<u>技术，以提升环境光照的估计精度</u>。  
• <u>通过同时建模表面相关属性和光照条件，在反射物体重建任务上达到了当前最优性能水平</u>。

<details class="lake-collapse"><summary id="u8596440f"><em><span class="ne-text">基于物理的</span></em><span class="ne-text">：引入物体表面反照率（albedo）、金属度（metallic）以及粗糙度（roughness）解决不同角度下高反射物体的 </span><em><span class="ne-text">视角相关外观</span></em><span class="ne-text"> 的变化。详见 </span><span class="ne-text" style="color: #1DC0C9">1.3.4</span><span class="ne-text"> 节。</span></summary><p id="u53d05ba4" class="ne-p"><br></p></details>
<details class="lake-collapse"><summary id="u04627db6"><em><span class="ne-text">延迟着色（deferred shading）</span></em><span class="ne-text">：2DGS 默认的</span><span class="ne-text" style="text-decoration: underline">前向着色（Forward Shading）</span><span class="ne-text">边计算几何边计算光照，通常光线穿过多个高斯体进行颜色累积，但是每个高斯点的几何信息不同（如法线方向、漫反射方向等信息），对于高反射物体不友好；</span><span class="ne-text" style="text-decoration: underline">延迟着色（deferred shading）</span><span class="ne-text">将几何和光照计算分开，先统计几何，再进行着色。详见 </span><span class="ne-text" style="color: #1DC0C9">1.3.5 </span><span class="ne-text">节。</span></summary><p id="ua6c68daa" class="ne-p"><img src="https://cdn.nlark.com/yuque/0/2026/png/45861457/1771663381824-9fe701c7-2332-48c7-99bb-49e1f028b19d.png" width="839.7982924069223" title="" crop="0,0,1,1" id="u49f3a1e7" class="ne-image"></p></details>
## Related work
### Radiance field for geometry reconstruction（用于几何重建的辐射场）
<u>神经重建方法大致可以分为以下几类：神经辐射场（Neural Radiance Fields，NeRF）、有符号距离场（Signed Distance Fields，SDF）以及高斯点绘制（Gaussian Splatting，GS）</u>。首先，NeRF 作为一种开创性方法而广为人知，它基于多层感知机（MLP）实现了复杂场景的新视角合成。NeRF 的代表性变体包括用于抗锯齿的 Mip-NeRF _[2]_、用于快速渲染的 Plenoctrees _[49]_，以及用于快速重建的 Instant NeRF _[53]_。其次，SDF 方法由 NeuS _[53]_ 推广开来，该方法利用 MLP 对 SDF 进行编码，以建模场景的三维表面。近年来的 SDF 方法还包括利用多分辨率哈希编码实现快速神经表面重建的 NeuS2 _[45]_，以及通过高阶导数监督提升重建质量的 Neuralangelo _[30]_。最后，GS 由文献 _[26]_ 近期提出，作为一种在渲染速度和照片一致性方面表现突出的神经渲染替代方案。其后续扩展工作包括使用二维高斯替代三维高斯的 2DGS _[20]_、沿最小主轴对三维高斯进行扁平化处理的 PGSR _[6]_，以及融合 SDF 与 GS 各自优势的 GSDF _[50]_。

<details class="lake-collapse"><summary id="u9729151a"><em><span class="ne-text">[2, 49, 53]</span></em></summary><p id="ued2cc60e" class="ne-p"><span class="ne-text">[2] </span><strong><span class="ne-text">Mip-NeRF: A Multiscale Representation for Anti-aliasing Neural Radiance Fields</span></strong><span class="ne-text">（IEEE/CVF International Conference on Computer Vision, 2021）：提出 Mip-NeRF 多尺度辐射场表示方法，通过构建锥形采样与抗锯齿积分渲染机制解决经典 NeRF 在高频区域出现的模糊与锯齿问题，显著提升多分辨率场景下的渲染质量与几何连续性。</span></p><p id="u1861bae5" class="ne-p"><span class="ne-text">[49] </span><strong><span class="ne-text">PlenOctrees for Real-time Rendering of Neural Radiance Fields</span></strong><span class="ne-text">（IEEE/CVF International Conference on Computer Vision, 2021）：提出 PlenOctrees 方法，将神经辐射场知识蒸馏至球谐系数八叉树结构中，实现神经辐射场的实时高效渲染，在保证视觉质量的同时大幅降低推理计算复杂度，推动 NeRF 技术向实时应用方向发展。</span></p><p id="u7f014a28" class="ne-p"><span class="ne-text">[53] </span><strong><span class="ne-text">Instant-Nerf: Instant On-device Neural Radiance Field Training via Algorithm-Accelerator Co-designed Near-memory Processing</span></strong><span class="ne-text">（ACM/IEEE Design Automation Conference, 2023）：提出面向硬件协同设计的即时 NeRF 训练方案，通过近内存计算架构与算法-加速器联合优化实现端侧快速场景重建，使神经辐射场能够在资源受限设备上实现实时训练与部署。</span></p></details>
<details class="lake-collapse"><summary id="ubba4c593"><em><span class="ne-text">[53, 45, 30]</span></em></summary><p id="ua888c835" class="ne-p"><span class="ne-text">[53] </span><strong><span class="ne-text">Instant-Nerf: Instant On-device Neural Radiance Field Training via Algorithm-Accelerator Co-designed Near-memory Processing</span></strong><span class="ne-text">（ACM/IEEE Design Automation Conference, 2023）：提出面向硬件协同设计的即时 NeRF 训练方案，通过近内存计算架构与算法-加速器联合优化实现端侧快速场景重建，使神经辐射场能够在资源受限设备上实现实时训练与部署。</span></p><p id="uae33a24d" class="ne-p"><span class="ne-text">[45] </span><strong><span class="ne-text">NeuS2: Fast Learning of Neural Implicit Surfaces for Multi-view Reconstruction</span></strong><span class="ne-text">（IEEE/CVF International Conference on Computer Vision, 2023）：在 NeuS 的基础上提出加速隐式表面学习的多视图重建框架，通过改进符号距离场与体渲染损失的联合优化策略提升训练效率，使神经隐式曲面能够更快收敛并保持较高的几何重建质量。</span></p><p id="u4d428a37" class="ne-p"><span class="ne-text">[30] </span><strong><span class="ne-text">Neuralangelo: High-Fidelity Neural Surface Reconstruction</span></strong><span class="ne-text">（IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023）：提出高保真神经隐式表面重建方法，结合多尺度网格结构与自适应高频建模能力，实现复杂场景下的精细几何恢复，在细节纹理区域和高曲率结构上显著优于传统神经隐式重建模型。</span></p></details>
<details class="lake-collapse"><summary id="u1cd74f8d"><em><span class="ne-text">[26, 20, 6, 50]</span></em></summary><p id="u3cf9645b" class="ne-p"><span class="ne-text">[26] </span><strong><span class="ne-text">3D Gaussian Splatting for Real-time Radiance Field Rendering</span></strong><span class="ne-text">（ACM Transactions on Graphics, 2023）：首次系统性提出 3D Gaussian Splatting 显式场景表示方法，通过可优化的高斯原语替代神经隐式辐射场，实现高质量实时渲染，在保证视觉真实性的同时显著降低训练与推理计算开销，推动辐射场走向实时应用。</span></p><p id="ua94fe13f" class="ne-p"><span class="ne-text">[20] </span><strong><span class="ne-text">2D Gaussian Splatting for Geometrically Accurate Radiance Fields</span></strong><span class="ne-text">（SIGGRAPH, 2024）：提出 2D 高斯平面化表示方法，将三维体高斯投影为二维平面圆盘结构以增强几何约束能力，在提升表面重建精度的同时保持高效渲染性能，实现几何准确性与视觉质量的统一优化。</span></p><p id="uc35f725c" class="ne-p"><span class="ne-text">[6] </span><strong><span class="ne-text">PGSR: Planar-based Gaussian Splatting for Efficient and High-Fidelity Surface Reconstruction</span></strong><span class="ne-text">（arXiv, 2024）：提出基于平面结构先验的 Gaussian Splatting 表面重建框架，通过引入平面分割与几何约束策略提升复杂场景下的表面连续性与重建精度，兼顾计算效率与高保真重建效果。</span></p><p id="u89689050" class="ne-p"><span class="ne-text">[50] </span><strong><span class="ne-text">GSDF: 3DGS Meets SDF for Improved Rendering and Reconstruction</span></strong><span class="ne-text">（2024）：提出融合 3D Gaussian Splatting 与 Signed Distance Function 的混合重建方法，通过联合优化显式辐射表示与隐式几何约束，在提升渲染质量的同时增强表面几何一致性。</span></p></details>
### Reflective object reconstruction（反射物体重建）
<u>高反射物体的重建具有很大的挑战性，通常需要进行专门的处理</u>。文献 _[21]_ 所报道的传统透明与反射物体重建方法依赖于对光学现象和物理建模的深入理解，并且往往针对不同的具体情况分别进行建模。近年来，基于神经辐射场的方法使得对多种复杂光学现象以及大规模场景的建模更加高效可行。

<details class="lake-collapse"><summary id="ua6529185"><em><span class="ne-text">[21]</span></em></summary><p id="uc6af79fc" class="ne-p"><span class="ne-text">[21] </span><strong><span class="ne-text">Transparent and Specular Object Reconstruction</span></strong><span class="ne-text">（Computer Graphics Forum, 2010）：系统研究透明与高镜面物体的三维重建问题，探讨透射与反射光学特性对多视图重建的影响，为后续透明物体神经重建方法提供了理论与算法基础。</span></p></details>
在基于 NeRF 的方法中，NeRFReN_ [19]_ 通过引入第二个 MLP 分支来建模反射效应；Ref-NeRF _[42]_ 使用集成方向编码（Integrated Directional Encoding）替代了 NeRF 中的视角相关建模方式；Planar Reflection-Aware NeRF _[17]_ 则将平面反射体的识别与建模视为一个光线追踪问题。

<details class="lake-collapse"><summary id="ubab8293f"><em><span class="ne-text">[19, 42, 17]</span></em></summary><p id="ud1b4c4f3" class="ne-p"><span class="ne-text">[19] </span><strong><span class="ne-text">NeRFReN: Neural Radiance Fields with Reflections</span></strong><span class="ne-text">（IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022）：提出面向反射场景的神经辐射场重建方法，通过显式建模视依赖反射分量来刻画镜面高光与环境反射变化，提升复杂光照条件下场景外观的一致性与重建鲁棒性。</span></p><p id="ueda571d0" class="ne-p"><span class="ne-text">[42] </span><strong><span class="ne-text">Ref-NeRF: Structured View-dependent Appearance for Neural Radiance Fields</span></strong><span class="ne-text">（IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022）：提出结构化视依赖外观建模的 NeRF 改进方法，通过分解反射与漫反射外观分量提升复杂材质表面的渲染真实性，在镜面反射场景中表现出更强的建模能力。</span></p><p id="u33e65ef7" class="ne-p"><span class="ne-text">[17] </span><strong><span class="ne-text">Planar Reflection-aware Neural Radiance Fields</span></strong><span class="ne-text">（arXiv, 2024）：提出面向平面反射结构的神经辐射场建模方法，通过显式引入平面镜面反射先验来约束视依赖外观变化，提升场景中镜面反射区域的重建稳定性与渲染一致性，特别适用于具有规则反射面的复杂光照环境。</span></p></details>
在基于 SDF 的方法中，NeRO _[32]_ 针对高反射物体提出了一种基于狭缝求和近似（slit-sum approximation）的新型光照表示；TensorSDF _[28]_ 则采用了一种考虑表面粗糙度的张量化表示方式。

<details class="lake-collapse"><summary id="uc8669fc5"><em><span class="ne-text">[32, 28]</span></em></summary><p id="u7fd25f6c" class="ne-p"><span class="ne-text">[32] </span><strong><span class="ne-text">NeRO: Neural Geometry and BRDF Reconstruction of Reflective Objects from Multiview Images</span></strong><span class="ne-text">（ACM Transactions on Graphics, 2023）：提出用于高反射物体重建的神经几何与 BRDF 联合优化框架，通过显式建模表面反射物理属性实现几何结构与材质参数的同步恢复，在多视角观测条件下有效缓解反射歧义问题，显著提升复杂反射场景的重建精度与渲染真实性。<br /></span><span class="ne-text">[28] </span><strong><span class="ne-text">TensoSDF: Roughness-aware Tensorial Representation for Robust Geometry and Material Reconstruction</span></strong><span class="ne-text">（ACM Transactions on Graphics, 2024）：提出粗糙度感知的张量化 SDF 表示方法，通过构建几何与材质耦合的高维参数张量实现复杂反射表面的稳定建模，在强光照变化与高反射场景中提升重建的鲁棒性与几何–材质一致性。</span></p></details>
对于基于 GS 的方法，R3DG _[18]_ 提出了一种基于点的光线追踪策略，以提升高反射物体的重光照质量；GS-IR _[31]_ 与 GaussianShader _[22]_ 分别侧重于引入逆渲染机制以及对渲染方程的简化近似。

<details class="lake-collapse"><summary id="ufe237582"><em><span class="ne-text">[18, 31, 22]</span></em></summary><p id="u38b12b55" class="ne-p"><span class="ne-text">[18] </span><strong><span class="ne-text">Relightable 3D Gaussian: Real-time Point Cloud Relighting with BRDF Decomposition and Ray Tracing</span></strong><span class="ne-text">（arXiv, 2023）：提出可重光照 3D 高斯点云表示方法，通过 BRDF 分解结合光线追踪渲染机制实现实时点云重照明，在保证高效渲染的同时增强材质反射建模能力，适用于动态光照条件下的场景外观恢复。</span></p><p id="u889add6a" class="ne-p"><span class="ne-text">[31] </span><strong><span class="ne-text">GS-IR: 3D Gaussian Splatting for Inverse Rendering</span></strong><span class="ne-text">（IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024）：提出用于逆向渲染的高斯 Splatting 框架，通过联合优化几何形状、光照参数与材质属性实现物理一致的场景重建，在多视图反射场景中有效提升外观解耦能力与重建精度。</span></p><p id="uf93feaed" class="ne-p"><span class="ne-text">[22] </span><strong><span class="ne-text">GaussianShader: 3D Gaussian Splatting with Shading Functions for Reflective Surfaces</span></strong><span class="ne-text">（IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024）：提出引入可学习着色函数的高斯渲染模型，通过替代传统球谐表示来刻画复杂反射表面的光照响应，在高反射材质场景中显著改善渲染真实性与视觉一致性。</span></p></details>
### Monocular depth/normal estimation（单目深度 / 法线估计）
<u>单目深度估计旨在从单张 RGB 图像中预测场景深度</u> _[15, 39]_。<u>该任务的发展历程从早期的多尺度网络</u> _[14]_，<u>逐步演进到基于 Transformer 的先进方法</u> _[38, 46, 51]_ <u>以及扩散模型</u> _[33, 40]_。诸如 MiDaS _[37]_ 和 MegaDepth _[29]_ 等方法利用大规模数据集实现了仿射不变的深度预测，在未见过的场景上具有良好的泛化能力，但其在尺度和偏移上仍然存在不确定性，从而限制了其在_度量深度估计 _中的应用。基于扩散模型的方法（如 DiffusionDepth _[12]_ 和 VPD _[52]_）通过引入预训练的潜空间扩散模型，进一步提升了跨域泛化性能。<u>然而，这类方法通常需要进行微调或引入额外输入（如相机内参），从而带来了较高的计算开销</u>。

<details class="lake-collapse"><summary id="u70c16e83"><em><span class="ne-text">度量深度估计</span></em><span class="ne-text">：输出真实的物理距离，而非尺度和偏移量不确定的相对深度。</span></summary><p id="uedbb3752" class="ne-p"><img src="https://cdn.nlark.com/yuque/0/2026/png/45861457/1771665345493-816415b6-0e35-4d3d-88fd-4dcfef031708.png" width="830.1176204483411" title="" crop="0,0,1,1" id="ud2832cae" class="ne-image"></p></details>
<details class="lake-collapse"><summary id="uf20731be"><em><span class="ne-text">[15, 39]</span></em></summary><p id="u31fb3d0d" class="ne-p"><span class="ne-text">[15] </span><strong><span class="ne-text">Depth Map Prediction from a Single Image Using a Multi-scale Deep Network</span></strong><span class="ne-text">（Advances in Neural Information Processing Systems, 2014）：首次提出基于多尺度深度卷积网络的单目深度预测方法，采用粗到细的多分辨率特征学习策略提升单张 RGB 图像的深度估计精度，推动深度学习在三维几何感知任务中的应用。</span></p><p id="u3498b24d" class="ne-p"><span class="ne-text">[39] </span><strong><span class="ne-text">Make3D: Learning 3D Scene Structure from a Single Still Image</span></strong><span class="ne-text">（IEEE Transactions on Pattern Analysis and Machine Intelligence, 2008）：提出早期单目三维场景结构学习框架，通过统计学习与几何先验结合的方法从单张图像恢复深度与空间结构，是传统单目三维重建研究的重要里程碑工作。</span></p></details>
<details class="lake-collapse"><summary id="u37141667"><em><span class="ne-text">[14, 38, 46, 51, 33, 40]</span></em></summary><p id="uff0906be" class="ne-p"><span class="ne-text">[14] </span><strong><span class="ne-text">Predicting Depth, Surface Normals and Semantic Labels with a Common Multi-scale Convolutional Architecture</span></strong><span class="ne-text">（IEEE International Conference on Computer Vision, 2015）：提出统一的多任务多尺度卷积预测框架，利用共享特征表示同时估计深度、法线和语义信息，验证了多任务学习在几何感知预测中的有效性。</span></p><p id="u23cd7d83" class="ne-p"><span class="ne-text">[38] </span><strong><span class="ne-text">Vision Transformers for Dense Prediction</span></strong><span class="ne-text">（IEEE/CVF International Conference on Computer Vision, 2021）：首次将视觉 Transformer 引入密集预测任务，通过全局自注意力机制增强远距离像素依赖建模能力，在单目深度估计与结构预测任务中显著提升空间一致性。<br /></span><span class="ne-text">[46] </span><strong><span class="ne-text">Transformer-based Attention Networks for Continuous Pixel-wise Prediction</span></strong><span class="ne-text">（IEEE/CVF International Conference on Computer Vision, 2021）：提出基于注意力机制的连续像素回归网络结构，通过强化特征交互与上下文建模能力提升逐像素几何属性预测精度，适用于深度与表面属性的密集回归任务。<br /></span><span class="ne-text">[51] </span><strong><span class="ne-text">MonoViT: Self-supervised Monocular Depth Estimation with a Vision Transformer</span></strong><span class="ne-text">（International Conference on 3D Vision, 2022）：提出基于自监督学习的 Transformer 单目深度估计模型，利用视觉注意力结构提升无监督深度学习中的全局结构恢复能力，在降低标注依赖的同时提高深度预测质量。</span></p><p id="u7c89b313" class="ne-p"><span class="ne-text">[33] </span><strong><span class="ne-text">Stealing Stable Diffusion Prior for Robust Monocular Depth Estimation</span></strong><span class="ne-text">（arXiv, 2024）：探索利用扩散模型中的结构先验提升单目深度估计鲁棒性，通过迁移稳定扩散模型的潜在几何特征增强深度预测的边缘一致性与全局结构恢复能力，在弱纹理与复杂遮挡场景下表现出较强泛化性能。<br /></span><span class="ne-text">[40] </span><strong><span class="ne-text">The Surprising Effectiveness of Diffusion Models for Optical Flow and Monocular Depth Estimation</span></strong><span class="ne-text">（Advances in Neural Information Processing Systems, 2024）：系统研究扩散模型在光流与单目深度预测任务中的潜在能力，发现预训练生成扩散模型能够隐式编码场景几何结构信息，在无需复杂任务专用设计的情况下即可实现较高精度的密集几何估计。</span></p></details>
<details class="lake-collapse"><summary id="u8d2dc032"><em><span class="ne-text">[37, 29]</span></em></summary><p id="u57628b2c" class="ne-p"><span class="ne-text">[37] </span><strong><span class="ne-text">Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer</span></strong><span class="ne-text">（IEEE Transactions on Pattern Analysis and Machine Intelligence, 2020）：提出多数据集混合训练策略以提升单目深度估计的跨域泛化能力，通过零样本跨数据迁移优化深度预测模型的稳定性，使模型能够在未见场景分布下仍保持较高的深度估计精度。</span></p><p id="u908e08da" class="ne-p"><span class="ne-text">[29] </span><strong><span class="ne-text">MegaDepth: Learning Single-view Depth Prediction from Internet Photos</span></strong><span class="ne-text">（IEEE Conference on Computer Vision and Pattern Recognition, 2018）：提出基于互联网多视角照片的大规模弱监督单目深度学习方法，利用结构化运动恢复与大规模图像对构建深度训练数据集，显著提升单图深度预测模型的几何恢复能力与实际场景适应性。</span></p></details>
<details class="lake-collapse"><summary id="u1fade05b"><em><span class="ne-text">[12, 52]</span></em></summary><p id="u691e8895" class="ne-p"><span class="ne-text">[12] </span><strong><span class="ne-text">DiffusionDepth: Diffusion Denoising Approach for Monocular Depth Estimation</span></strong><span class="ne-text">（European Conference on Computer Vision, 2025）：提出基于扩散去噪过程的单目深度估计方法，通过利用扩散模型的结构先验与迭代去噪机制恢复高质量深度图，在复杂纹理缺失与遮挡区域提升深度预测的鲁棒性与精度。</span></p><p id="ue33ca659" class="ne-p"><span class="ne-text">[52] </span><strong><span class="ne-text">Unleashing Text-to-Image Diffusion Models for Visual Perception</span></strong><span class="ne-text">（IEEE/CVF International Conference on Computer Vision, 2023）：探索文本到图像扩散模型在视觉感知任务中的迁移能力，通过挖掘生成模型中的结构表征潜力，将预训练扩散模型应用于几何感知与场景理解任务，拓展生成式模型在计算机视觉中的应用范围。</span></p></details>
<u>相比之下，表面法线估计</u> _[52]_ <u>能够在不引入度量歧义的情况下捕获局部几何细节，因此在场景重建和目标定位等任务中具有重要价值</u>。诸如 GeoNet _[48]_ 和 DSINE _[1]_ 等方法通过引入深度–法线一致性约束并利用多样化的数据集来提升泛化能力，而 OmniData_ [13]_ 则提供了大规模标注的法线数据以增强模型训练效果。<u>这些方法的共同目标在于提升泛化性能、增强遮挡边界的清晰度，并减少对大规模标注数据集的依赖，为构建更加鲁棒且可扩展的解决方案提供了途径</u>。**本文正是基于上述研究进展，探索如何利用单目图像中的深度与法线信息来提升反射物体的重建效果**。

<details class="lake-collapse"><summary id="uf4888506"><em><span class="ne-text">[52, 48, 1, 13]</span></em></summary><p id="u81084875" class="ne-p"><span class="ne-text">[52] </span><strong><span class="ne-text">Unleashing Text-to-Image Diffusion Models for Visual Perception</span></strong><span class="ne-text">（IEEE/CVF International Conference on Computer Vision, 2023）：探索文本到图像扩散模型在视觉感知任务中的迁移能力，通过挖掘生成模型中的结构表征潜力，将预训练扩散模型应用于几何感知与场景理解任务，拓展生成式模型在计算机视觉中的应用范围。</span></p><p id="u643dc4ce" class="ne-p"><span class="ne-text">[48] </span><strong><span class="ne-text">GeoNet: Unsupervised Learning of Dense Depth, Optical Flow and Camera Pose</span></strong><span class="ne-text">（IEEE Conference on Computer Vision and Pattern Recognition, 2018）：提出联合无监督学习深度、光流与相机位姿的多任务几何学习框架，通过几何一致性约束实现跨模态几何信息协同优化，在无需真实标注的情况下提升场景运动与结构恢复能力。</span></p><p id="uef63ddb6" class="ne-p"><span class="ne-text">[1] </span><strong><span class="ne-text">Rethinking Inductive Biases for Surface Normal Estimation</span></strong><span class="ne-text">（IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024）：系统研究表面法线估计中的归纳偏置设计问题，通过优化网络结构与特征建模策略增强几何结构感知能力，在表面方向预测任务中显著提升细节恢复与稳定性。</span></p><p id="ud6d16534" class="ne-p"><span class="ne-text">[13] </span><strong><span class="ne-text">OmniData: A Scalable Pipeline for Making Multitask Mid-level Vision Datasets from 3D Scans</span></strong><span class="ne-text">（IEEE/CVF International Conference on Computer Vision, 2021）：提出基于三维扫描数据的大规模多任务中层视觉数据生成流程，通过自动标注与合成监督信号构建统一的几何感知训练数据集，促进深度、法线及语义等多视觉任务的联合学习。</span></p></details>
## Method
<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/45861457/1769411194356-7f266ddc-480f-45a1-ad16-b34278c9f039.png)

**图 1**. 我们提出的 GS-2DSG 反射物体重建方法整体框架。基础模型通过估计的深度图和法线图对 2DGS 重建进行监督，以提升几何精度与表面平滑性；同时学习并渲染环境贴图以刻画反射效果。

### Preliminary: 2D Gaussian Splatting（预备知识：二维高斯点绘制）
由于我们关注于实现高质量的重建效果，<u>本文的方法基于当前性能领先的、以 </u>_<u>surfel</u>_<u> 为基础的 2DGS</u>_ [20]_，该方法在几何表达能力和计算效率方面均表现优异。<u>2DGS</u> 提出<u>将三维体积表示压缩为一组具有二维方向性的平面高斯圆盘</u>，并引入了一种具备透视精度的二维点绘制（2D splatting）过程。<u>与 3DGS 类似，每一个二维高斯点由其中心点</u>$ \mathbf{p}_k $<u>、两条主切向量</u>$ \mathbf{t}_u $<u>和</u>$ \mathbf{t}_v $<u>以及对应的尺度向量</u>$ \mathbf{S} = (s_u, s_v) $<u>所表示</u>，其中尺度向量用于控制二维高斯分布在两个主方向上的方差。<u>相比于 3DGS，2DGS 能够更好地表示场景几何结构，这是因为具有方向性的平面高斯可以与真实表面实现精确对齐，同时其法线方向也能够自然地定义为该平面的法向量</u>。

<details class="lake-collapse"><summary id="u5868ce50"><em><span class="ne-text">surfel (Surface + Element) </span></em><span class="ne-text">： 可以理解为 一个带有几何属性的“带方向的小圆片”，通常包含位置、法向量、半径/尺度等信息。    </span></summary><p id="ue89ae4e0" class="ne-p"><img src="https://cdn.nlark.com/yuque/0/2026/png/45861457/1771746433315-fdaca762-51f5-42eb-90e3-2342bfa49413.png" width="877.7142575780321" title="" crop="0,0,1,1" id="ub1abd166" class="ne-image"></p></details>
<details class="lake-collapse"><summary id="ud2f5cfd1"><em><span class="ne-text">[20]</span></em></summary><p id="u90a5e781" class="ne-p"><span class="ne-text">[20] </span><strong><span class="ne-text">2D Gaussian Splatting for Geometrically Accurate Radiance Fields</span></strong><span class="ne-text">（SIGGRAPH, 2024）：提出 2D 高斯平面化表示方法，将三维体高斯投影为二维平面圆盘结构以增强几何约束能力，在提升表面重建精度的同时保持高效渲染性能，实现几何准确性与视觉质量的统一优化。</span></p></details>
具体而言，一个<u>二维高斯</u>圆盘是在世界坐标系中的局部切平面内<u>定义</u>的，其参数化形式为：

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/45861457/1769409453167-74795022-6ba2-409b-bd96-17a2a58daf22.png)

> 这里的$ \mathbf{x} $应该和前文中$ \mathbf{p}_k $相同，表示二维高斯中心点。
>

<details class="lake-collapse"><summary id="u18e0ec56"><em><span class="ne-text">公式（1）</span></em><span class="ne-text">：二维高斯“圆盘”定义，两条主切向量构成局部坐标系，尺度向量“圆盘”拉长还是扁平。</span></summary><p id="u87c0c571" class="ne-p"><span class="ne-text">给定二维参数</span><span id="C0Z8I" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/838c98c94a4322ee88230f6f92e2ac73.svg"></span><span class="ne-text">，我们可以得到该二维高斯圆盘上任意一点</span><span id="i578L" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/02d7b635fdcaa90d77505f57938270a9.svg"></span><span class="ne-text">在三维空间中的位置。  </span></p><p id="ud27669a3" class="ne-p"><img src="https://cdn.nlark.com/yuque/0/2026/png/45861457/1771747032821-65ed2e4b-5562-4b5d-ba79-b923d23dd4e4.png" width="794.6218232668765" title="" crop="0,0,1,1" id="u494ce7de" class="ne-image"></p></details>
对于位于$ uv $空间中的点$ \mathbf{u}(u, v) $，其对应的<u>二维高斯值</u>可以通过标准高斯函数<u>计算</u>得到：

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/45861457/1769409476895-543e1a7e-2da5-449b-bbad-6593fdfa2a2a.png)

其中，中心点$ \mathbf{x} $、尺度参数$ (s_u, s_v) $以及旋转参数$ (\mathbf{t}_u, \mathbf{t}_v) $均为可学习参数。与 3DGS _[26]_ 一致，每一个二维高斯基元还具有不透明度$ \alpha $，以及由球谐函数参数化的_视角相关外观_$ c $。<u>不同于将</u>_<u>二维高斯基元 </u>_<u>直接投影到图像空间进行渲染的方式，</u>_<u>2DGS</u>_<u> 通过平面求交的方式，在局部切平面中计算光线与高斯圆盘的交点，从而有效缓解了高斯点在</u>_<u>掠射角 </u>_<u>条件下发生退化的问题</u>。

<details class="lake-collapse"><summary id="u11e93440"><em><span class="ne-text">公式（2）</span></em><span class="ne-text">：公式（1）决定二维高斯点位置和形状，公式（2）决定其在平面内的密度权重。</span></summary><p id="u2f765522" class="ne-p"><img src="https://cdn.nlark.com/yuque/0/2026/png/45861457/1771747388311-f664d97f-c935-419b-ac84-9c997627f629.png" width="812.3697218576087" title="" crop="0,0,1,1" id="u784588a7" class="ne-image"></p></details>
<details class="lake-collapse"><summary id="ud622b303"><em><span class="ne-text" style="text-decoration: line-through">视角相关外观 c</span></em><span class="ne-text" style="text-decoration: line-through">：这里没直接说 c 是 RGB 颜色（原 3DGS 中该参数表示颜色），是因为作者用 2DGS + PBR，还用到了物体表面反照率（albedo）、金属度（metallic）以及粗糙度（roughness）来表示其外观。</span></summary><p id="u5c585018" class="ne-p"><br></p></details>
<details class="lake-collapse"><summary id="u184f1d7b"><em><span class="ne-text">二维高斯基元 &amp;&amp; 2DGS</span></em><span class="ne-text">：前者是 3DGS 基于投影在二维空间中的数学对象，后者是 基于局部切平面二维高斯圆盘、通过光线-平面求交进行渲染的几何表示与渲染框架。  </span></summary><p id="u44be2851" class="ne-p"><img src="https://cdn.nlark.com/yuque/0/2026/png/45861457/1771748204382-ca293ebd-3112-49d6-8b67-784239c5ca43.png" width="804.3024952254577" title="" crop="0,0,1,1" id="ue857c592" class="ne-image"></p></details>
<details class="lake-collapse"><summary id="u6cfeab9c"><em><span class="ne-text">琼射角（grazing angle）</span></em><em><span class="ne-text" style="color: #74B602">退化问题</span></em><span class="ne-text">：当视线方向与物体表面几乎平行 时的观察角度。此时 法线方向 ⟂ 视线方向，3DGS 会得到一个特别扁的椭圆，但实际上应该是一个面。</span></summary><p id="u0f6b09f7" class="ne-p"><img src="https://cdn.nlark.com/yuque/0/2026/png/45861457/1771747645384-acaecd23-05b2-4a4a-abc6-cda3e5e06bae.png" width="793.0083779404463" title="" crop="0,0,1,1" id="u9458fc8f" class="ne-image"></p></details>
<details class="lake-collapse"><summary id="uac5d7a98"><em><span class="ne-text">[26]</span></em></summary><p id="u99e1598d" class="ne-p"><span class="ne-text">[26] </span><strong><span class="ne-text">3D Gaussian Splatting for Real-time Radiance Field Rendering</span></strong><span class="ne-text">（ACM Transactions on Graphics, 2023）：首次系统性提出 3D Gaussian Splatting 显式场景表示方法，通过可优化的高斯原语替代神经隐式辐射场，实现高质量实时渲染，在保证视觉真实性的同时显著降低训练与推理计算开销，推动辐射场走向实时应用。</span></p></details>
其光栅化过程与 3DGS 类似。首先根据高斯中心点的深度对二维高斯进行排序，然后采用体渲染中的 alpha 混合方式，从前到后累积由 alpha 加权的外观信息：

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/45861457/1769409501186-989a9bcf-b3f7-47ae-8ad4-239a0636f444.png)

其中，$ \hat{G} $表示文献 [4] 中引入的物体空间低通滤波器。更多细节可参考原始论文。

<details class="lake-collapse"><summary id="u564fdb65"><em><span class="ne-text">公式（3）</span></em><span class="ne-text">：计算像素颜色。</span></summary><p id="u3bf279fb" class="ne-p"><img src="https://cdn.nlark.com/yuque/0/2026/png/45861457/1771749259542-1ed696cf-24f9-4c7f-a0f5-37693b489d05.png" width="798.655436582952" title="" crop="0,0,1,1" id="u6e86f202" class="ne-image"></p></details>
<details class="lake-collapse"><summary id="uc1ba84f7"><em><span class="ne-text">[4]</span></em></summary><p id="uae0013ac" class="ne-p"><span class="ne-text">[4] High-Quality Surface Splatting on Today’s GPUs（Eurographics/IEEE VGTC Symposium on Point-Based Graphics 2005）：提出一种面向 GPU 实时渲染的高质量表面 Splatting 方法，通过改进的屏幕空间滤波与重建策略，实现了高效、平滑且抗锯齿的点基表面渲染效果，为后续基于点表示的实时重建与渲染方法（如 3D Gaussian Splatting）奠定了重要的图形学基础。</span></p></details>
### Geometric properties from Gaussian（来自高斯表示的几何属性）
<u>为重建物体表面并利用基础模型所提供的几何信息，我们从二维高斯基元中推导出深度与法线的渲染方式</u>。

#### Normal（法线）
不同于其他基于 3DGS 的方法 _[7, 8]_ 通常将尺度最小的轴方向假设为法线方向，<u>2DGS 中的二维高斯基元具有明确且物理意义清晰的法线定义，即其所在平面的法向量</u>。具体而言，每一个二维高斯基元的法线可以通过两条主切向量的叉积计算得到：

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/45861457/1769409646674-75f2221d-827a-4225-90db-d0b70e3b244b.png)

<details class="lake-collapse"><summary id="u986c5e0d"><em><span class="ne-text">[7, 8]</span></em></summary><p id="uc8a18012" class="ne-p"><span class="ne-text">[7] NeuSG: Neural Implicit Surface Reconstruction with 3D Gaussian Splatting Guidance（arXiv 2023）：提出将 3D Gaussian Splatting 引入神经隐式表面重建框架中，通过高效的高斯表示为 SDF/隐式函数学习提供几何与密度引导，在保持高质量几何重建能力的同时显著提升训练与渲染效率，缓解传统神经隐式方法收敛慢、优化不稳定的问题。</span></p><p id="uc4dfda97" class="ne-p"><span class="ne-text">[8] VCR-Gaus: View Consistent Depth-Normal Regularizer for Gaussian Surface Reconstruction（arXiv 2024）：提出一种视角一致性的深度–法线联合正则项，用于约束高斯表面重建过程中的几何一致性，通过跨视角深度与法线耦合优化，有效减少高斯表面中的几何噪声与不连续现象，提升复杂场景下的表面细节恢复与结构稳定性。</span></p></details>
随后，<u>屏幕空间中某一点</u>$ \mathbf{x} $<u>处的法线向量</u>可以采用与[式 (3)](#u5ef9d208) 中颜色渲染类似的方式进行累积渲染：

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/45861457/1769409662833-f5d89d01-77fa-4127-a461-d59b0f1c0ddb.png)

为简化表述，在此及后续公式中我们省略了低通滤波项$ \hat{G} $。

<details class="lake-collapse"><summary id="uc51a0399"><em><span class="ne-text">公式（5）</span></em><span class="ne-text">：计算像素法线。</span></summary><p id="uf714c3d2" class="ne-p"><img src="https://cdn.nlark.com/yuque/0/2026/png/45861457/1771749651390-e6600040-df3c-4299-80e0-feeceae12d3e.png" width="813.1764445208239" title="" crop="0,0,1,1" id="u920f1bc6" class="ne-image"></p></details>
#### Depth（深度）
对于深度的计算，我们<u>遵循 2DGS 的做法，通过高斯基元与光线的交点深度来渲染期望深度</u>，其形式为：

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/45861457/1769409758899-049e8617-bae3-4f5b-8933-03637085f5b4.png)

<details class="lake-collapse"><summary id="ud60745e0"><em><span class="ne-text">公式 （6）</span></em><span class="ne-text">：沿射线方向的加权平均深度。</span></summary><p id="ua94e7120" class="ne-p"><img src="https://cdn.nlark.com/yuque/0/2026/png/45861457/1771819324910-c8c17617-3bf8-4671-8304-b83270395957.png" width="885.7814842101832" title="" crop="0,0,1,1" id="u3b714b15" class="ne-image"></p></details>
### Supervision from Foundation Models（来自基础模型的监督）
<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/45861457/1769411066185-b536d197-9745-40f0-8019-701f7164dc78.png)

**图 2**. 来自基础模型 Marigold _[25, 34]_ 与 Depth Pro _[3]_ 的法线与深度预测质量对比。

<u>正如我们在</u>[<u>引言</u>](#bjyuo)<u>中所讨论的，单目深度与法线估计模型能够基于海量训练数据中所蕴含的先验知识，从单张图像中推断几何信息，因此对反射表面问题并不敏感</u>。<u>本文采用此类模型 </u>_<u>[3, 34]</u>_<u> 为每一幅图像预测法线</u>$ \tilde{\mathbf{N}} $<u>和深度</u>$ \tilde{D} $。如_图 2_ 所示，基础模型能够针对具有不同材质属性的物体给出较为可靠的预测结果。<u>我们将预测得到的法线</u>$ \tilde{\mathbf{N}}_i $<u>和深度</u>$ \tilde{D}_i $<u>作为伪真值（pseudo ground-truth），用于引入额外的监督信号</u>。

<details class="lake-collapse"><summary id="uded029e5"><em><span class="ne-text">[3, 25, 34]</span></em><span class="ne-text">：基础模型 Marigold 和 Depth Pro。</span></summary><p id="u4ebdc37a" class="ne-p"><span class="ne-text">[3] Depth Pro: Sharp Monocular Metric Depth in Less Than a Second（arXiv 2024）：提出一种高效的单目度量深度预测模型，采用高分辨率特征建模与优化的推理结构，在保证深度边界锐利度的同时实现接近实时的深度估计速度，显著提升了复杂场景下单目深度恢复的精度与实用性。</span></p><p id="ufceef15f" class="ne-p"><span class="ne-text">[25] Repurposing Diffusion-based Image Generators for Monocular Depth Estimation（CVPR 2024）：提出将预训练的扩散图像生成模型重新用于单目深度估计任务，通过利用扩散模型内部蕴含的语义与结构先验信息，实现高质量深度预测，在小样本条件下仍能保持较强的深度恢复能力。</span></p><p id="u5d1cd198" class="ne-p"><span class="ne-text">[34] Fine-tuning Image-conditional Diffusion Models Is Easier Than You Think（WACV 2025）：针对条件扩散模型微调过程复杂、计算开销大的问题，提出一种简单高效的微调策略，使图像条件扩散模型在较小计算资源下即可实现稳定收敛，并保持较高的生成质量与泛化能力。</span></p></details>
具体而言，<u>在法线监督方面，我们采用</u>$ L_1 $<u>损失与余弦相似度损失共同约束渲染得到的法线图</u>，其形式为：

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/45861457/1769409852562-0d6032ad-467e-431a-be97-f6815c56b511.png)

> $ \hat{} $为渲染值，$ \tilde{} $为基础模型得到的伪真值。
>

<u>在深度监督方面，我们采用</u>文献<u> </u>_[37]_ 中提出的<u>尺度不变深度损失</u>，对渲染得到的深度图进行约束，其定义为：

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/45861457/1769409866908-ca7d8395-4b1d-4517-9aa6-da548a314175.png)

其中，$ \omega $和$ b $分别表示用于对齐渲染深度$ \hat{D} $与预测深度$ \tilde{D} $的尺度因子与偏移量。根据文献 _[37]_ 的方法，我们通过最小二乘优化来求解$ \omega $和$ b $。

<details class="lake-collapse"><summary id="uf8444652"><em><span class="ne-text">[37]</span></em><em><span class="ne-text" style="color: #117CEE">*</span></em></summary><p id="u10d0b1b3" class="ne-p"><span class="ne-text">[37] Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer（TPAMI 2020）：提出通过多数据集混合训练策略提升单目深度估计模型的跨域泛化能力，实现零样本跨数据集迁移性能的显著改善，使深度预测在不同场景分布下仍保持较高的鲁棒性与稳定性。</span></p></details>
### Shading the Gaussians（高斯的着色）
#### Recap on Rendering Equation（渲染方程回顾）
<u>在原始的 Gaussian Splatting 中，每一个高斯基元的颜色由基于球谐函数参数化的视角相关外观</u>$ c $<u>表示。然而，对于反射表面而言，其外观特性难以通过球谐函数进行准确描述</u>。借鉴已有工作 _[18, 22, 31]_，我们引入基于物理的渲染（Physical-Based Rendering，PBR）管线，并利用<u>经典渲染方程</u> _[23]_ 来刻画空间点$ \mathbf{x} $在观察方向$ \boldsymbol{\omega}_o $下的出射辐射度，其形式为：

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/45861457/1769410034435-de608f35-45ca-4881-902f-36a50491ee7f.png)

其中，$ \Omega $表示以$ \mathbf{x} $为中心的上半球，$ \mathbf{n} $为局部表面的法线方向，$ L_i $表示来自方向$ \boldsymbol{\omega}_i $的入射辐射度，而$ f_r $为双向反射分布函数（Bidirectional Reflectance Distribution Function，BRDF）。

> 对于$ \boldsymbol{\omega}_i, \boldsymbol{\omega}_o $，可以理解为 所有方向来的光（$ \boldsymbol{\omega}_i $）经过 BRDF 反射汇聚成朝相机方向（$ \boldsymbol{\omega}_o $）的出射光。
>

<details class="lake-collapse"><summary id="ubffbff19"><em><span class="ne-text">[18, 22, 31]</span></em></summary><p id="u212fc7f8" class="ne-p"><span class="ne-text">[18] Relightable 3D Gaussian（arXiv 2023）：提出一种基于 BRDF 分解与光线追踪的可重光照 3D 高斯表示方法，通过显式建模材料反射特性，实现点云级别的实时重光照渲染，使高反射复杂表面场景下的光照恢复更加物理合理。</span></p><p id="uaab4e1b1" class="ne-p"><span class="ne-text">[22] GaussianShader: 3D Gaussian Splatting with Shading Functions for Reflective Surfaces（CVPR 2024）：提出将可学习的着色函数引入 3D Gaussian Splatting 框架，通过显式建模反射表面的视角依赖光照变化，增强高光与镜面反射区域的渲染质量，改善传统高斯表示对复杂材质的建模能力。</span></p><p id="ueeb0be06" class="ne-p"><span class="ne-text">[31] GS-IR: 3D Gaussian Splatting for Inverse Rendering（CVPR 2024）：提出基于高斯溅射的逆渲染框架，通过联合优化几何、材质与光照参数实现场景物理属性恢复，显著提升复杂光照条件下的真实感渲染与反射结构重建性能。</span></p></details>
<details class="lake-collapse"><summary id="u1bb088ba"><em><span class="ne-text">[23]</span></em><em><span class="ne-text" style="color: #117CEE">*</span></em></summary><p id="uada36fb4" class="ne-p"><span class="ne-text">[23] The Rendering Equation（SIGGRAPH 1986）：由 James T. Kajiya 在 SIGGRAPH 1986 上提出，首次形式化给出了</span><strong><span class="ne-text">渲染方程（Rendering Equation）</span></strong><span class="ne-text">，以积分形式统一描述光在场景中的入射、反射与出射过程，建立了全局光照的物理理论基础，为后续路径追踪（Path Tracing）、光子映射（Photon Mapping）等物理真实感渲染方法奠定了理论框架，是现代物理渲染（PBR）的核心理论起点。</span></p></details>
根据文献 _[5]_，<u>BRDF</u>$ f_r $可以分解为漫反射项和镜面反射项，并可进一步按照 Cook–Torrance BRDF 模型 _[9, 43]_ 表达为：

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/45861457/1769410064614-ffabe979-4ba6-4965-a6ef-0764c17e7afc.png)

其中，$ \mathbf{a} $、$ m $和$ \rho $分别表示表面的反照率（albedo）、金属度（metallic）以及粗糙度（roughness）；$ D $为法线分布函数（Normal Distribution Function），$ F $为菲涅耳项（Fresnel term），$ G $为几何遮挡项（Geometry term）。$ D $、$ F $和$ G $的具体计算方式均与表面属性相关，可参考文献 _[43]_。

<details class="lake-collapse"><summary id="uf9fe4593"><em><span class="ne-text">[5]</span></em><em><span class="ne-text" style="color: #117CEE">*</span></em></summary><p id="u6d20a793" class="ne-p"><span class="ne-text">[5] Physically-Based Shading at Disney（ACM SIGGRAPH 2012）：由 Brent Burley 在 Walt Disney Animation Studios 工作期间发表于 SIGGRAPH 2012，提出了统一、艺术家友好的 Disney BRDF 模型，在物理合理性的前提下兼顾可控性与参数直观性，系统整合了漫反射、镜面反射、次表面散射等项，成为工业界物理渲染（PBR）材质建模的标准范式，并广泛应用于电影级渲染流程。</span></p></details>
<details class="lake-collapse"><summary id="uaf0722a1"><em><span class="ne-text">[9, 43]</span></em></summary><p id="u5c4b5276" class="ne-p"><span class="ne-text">[9] A Reflectance Model for Computer Graphics（ACM SIGGRAPH 1981）：由 Robert L. Cook 与 Kenneth E. Torrance 在 SIGGRAPH 1981 提出经典的 </span><strong><span class="ne-text">Cook–Torrance 微表面反射模型</span></strong><span class="ne-text">，首次将微表面统计理论引入计算机图形学，通过法线分布函数（NDF）、几何遮蔽项（Geometry Term）与菲涅耳项（Fresnel Term）构建物理一致的镜面反射模型，为后续基于物理的渲染（PBR）奠定核心反射理论基础。</span></p><p id="uc85a0109" class="ne-p"><span class="ne-text">[43] Microfacet Models for Refraction through Rough Surfaces（EGSR 2007）：由 Bruce Walter、Stephen R. Marschner、Hongsong Li 与 Kenneth E. Torrance 在 Eurographics Symposium on Rendering 2007 提出，将微表面理论从反射扩展到</span><strong><span class="ne-text">粗糙表面的折射建模</span></strong><span class="ne-text">，建立了统一的微表面透射模型（Microfacet Refraction Model），系统推导了能量守恒形式的折射项，为玻璃、液体等透明材质的物理真实感渲染提供了理论支撑，并成为现代 PBR 管线中的标准透射模型基础。</span></p></details>
<u>根据</u><u><font style="color:#1DC0C9;">式 (10)</font></u><u>，</u><u><font style="color:#1DC0C9;">渲染方程式 (9) </font></u><u>可以重写为漫反射项</u>$ L_d $<u>与镜面反射项</u>$ L_s $<u>之和</u>，即：

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/45861457/1769410099776-c5fb47d6-ca65-429c-b436-ff8acb5bd2ab.png)

<u>我们遵循文献 </u>_<u>[36]</u>_<u>，使用一个可训练的高动态范围（High Dynamic Range，HDR）立方体贴图来表示</u>**<u>环境光照</u>**$ L_i(\boldsymbol{\omega}_i) $。其中，<u>漫反射项</u>$ L_d $<u>可以预先计算并存储为二维纹理贴图</u>；而对于<u>镜面反射项</u>$ L_s $，我们<u>采用分裂求和近似（split-sum approximation）</u>_<u>[24]</u>_<u> 对积分过程进行简化</u>。

<details class="lake-collapse"><summary id="ua84382c6"><em><span class="ne-text">[36, 24]</span></em><em><span class="ne-text" style="color: #117CEE">*</span></em></summary><p id="u9766f33c" class="ne-p"><span class="ne-text">[36] Extracting Triangular 3D Models, Materials, and Lighting from Images（CVPR 2022）：由 Jacob Munkberg、Jon Hasselgren、Tianchang Shen、Jun Gao、Wenzheng Chen、Alex Evans、Thomas Müller 与 Sanja Fidler 在 CVPR 2022 提出一种端到端可微渲染框架，可从多视图图像中同时恢复</span><strong><span class="ne-text">三角网格几何、PBR 材质参数与环境光照</span></strong><span class="ne-text">，实现可编辑、可重光照的高质量 3D 重建，推动了神经渲染向显式可控图形表示的融合发展。</span></p><p id="u1cae702f" class="ne-p"><span class="ne-text">[24] Real Shading in Unreal Engine 4（SIGGRAPH 2013 Course）：由 Brian Karis 在 Epic Games 工作期间于 SIGGRAPH 2013 课程报告中提出，系统构建了 Unreal Engine 4 的实时 PBR 着色模型，包括改进的微表面 BRDF、能量守恒处理及环境光照近似方案，实现了物理一致性与实时性能之间的工程平衡，成为实时图形渲染领域的工业标准方案之一。</span></p></details>
### Forward Shading vs. Deferred Shading（前向着色 vs. 延迟着色）
<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/45861457/1769411325538-81800304-2b64-414d-85b6-df4dbac1da17.png)

**图 3**. 前向着色（Forward Shading）与延迟着色（Deferred Shading）的对比。

<details class="lake-collapse"><summary id="ue86f1921"><em><span class="ne-text" style="font-size: 14px">图 3</span></em><span class="ne-text" style="font-size: 14px">：前向着色 每个高斯基元法线不一致，且每个高斯基元颜色都需要参与计算；延迟着色 先存储相关信息，再得到加权混合后的法线，每个像素仅计算一次颜色。</span></summary><p id="u8ee24c51" class="ne-p"><img src="https://cdn.nlark.com/yuque/0/2026/png/45861457/1771821188095-ed3f6884-37a3-4155-80ea-906680288903.png" width="889.0083748630435" title="" crop="0,0,1,1" id="uc9611182" class="ne-image"></p></details>
<u>为了将二维高斯基元与基于物理的渲染（PBR）管线相结合，我们为每一个高斯基元额外引入一组 PBR 参数</u> $ (a, m, \rho) $。<u>一种直接的做法是采用前向着色（forward shading）对二维高斯基元进行渲染，即根据</u><u><font style="color:#1DC0C9;">式 (11) </font></u><u>计算每一个高斯在位置</u>$ \mathbf{x} $<u>、观察方向</u>$ \boldsymbol{\omega}_o $<u>和法线</u>$ \mathbf{n} $<u>下的出射辐射度</u>$ L_o(\mathbf{x}, \boldsymbol{\omega}_o, \mathbf{n}) $<u>，并最终通过</u><u><font style="color:#1DC0C9;">式 (3)</font></u><u> 中的 alpha 混合方式得到累积颜色</u>。<u>然而，该方案存在以下几个明显缺陷</u>：

(1) <u>从</u><u><font style="color:#1DC0C9;">式 (11)</font></u><u> 可以看出，表面点的颜色由着色点位置</u>$ \mathbf{x} $<u>、法线方向</u>$ \mathbf{n} $<u>、观察方向</u>$ \boldsymbol{\omega}_o $<u> 以及 PBR 参数共同决定</u>。<u>在前向着色过程中</u>，如_图 3_ 所示，沿着同一条相机光线的所有高斯基元都会参与颜色累积。这会带来不准确性，因为<u>不同高斯基元往往具有不同的法线方向和空间位置，且并不一定都与真实表面的法线方向对齐</u>。

(2) <u>前向着色的计算开销较大</u>，因为<u>需要对沿光线的每一个高斯基元分别执行一次完整的着色计算</u>。

<u>基于上述分析，我们选择采用延迟着色（deferred shading）技术来对高斯基元进行渲染</u>。<u>延迟着色 </u>_<u>[11]</u>_<u> 是一种将着色与光照计算从几何渲染中解耦的渲染方法</u>。Ye 等人 [47] 首次将其引入 Gaussian Splatting 框架，以提升在存在镜面反射情况下的渲染质量。<u>该渲染流程通常分为两个阶段：几何渲染阶段与着色阶段</u>。

<u>在几何渲染阶段，我们将场景的深度、法线以及 PBR 参数渲染并存储到 G-buffer 中</u>；<u>在随后的着色阶段，仅基于 G-buffer 中的信息对每一条光线执行一次着色计算。如</u>_<u>图 3</u>_<u> 所示</u>，<u>延迟着色使得我们能够在正确的着色点位置及其对应的法线方向上进行光照计算，从而获得更加准确和稳定的着色结果</u>。

<details class="lake-collapse"><summary id="u17db6c42"><em><span class="ne-text">[11]</span></em></summary><p id="u4beb2333" class="ne-p"><span class="ne-text">[11] The Triangle Processor and Normal Vector Shader: A VLSI System for High Performance Graphics（ACM SIGGRAPH 1988）：由 Michael Deering、Stephanie Winner、Bic Schediwy、Chris Duffy 与 Neil Hunt 在 SIGGRAPH 1988 提出，设计了一种专用 VLSI 图形硬件架构，将</span><strong><span class="ne-text">三角形光栅化与法向量着色计算</span></strong><span class="ne-text">集成到硬件流水线中，实现了高性能实时图形处理。该工作奠定了现代 GPU 可编程图形管线的早期雏形，对后续顶点着色器与片元着色器架构的发展具有重要启发意义。</span></p></details>
### Training Objective（训练目标）
<u>我们采用多种损失函数对模型进行联合训练</u>。<u>首先，我们沿用 2DGS </u>_<u>[20]</u>_<u> 中的基础损失</u>$ \mathcal{L}_{\mathrm{GS}} $，<u>其中包括 RGB 重建损失和法线一致性损失</u>。<u>其次，引入第 </u>[<u>1.3.3 节</u>](#Xma2B)<u>中所提出的、来自基础模型的几何监督损失</u>$ \mathcal{L}_n $<u>和</u>$ \mathcal{L}_d $。<u>此外，为了促进训练过程的稳定性与收敛性，我们还对光照参数和 PBR 参数引入了正则化项</u>。

<details class="lake-collapse"><summary id="u6cfb89ac"><em><span class="ne-text">[20]</span></em></summary><p id="ubfca2513" class="ne-p"><span class="ne-text">[20] </span><strong><span class="ne-text">2D Gaussian Splatting for Geometrically Accurate Radiance Fields</span></strong><span class="ne-text">（SIGGRAPH, 2024）：提出 2D 高斯平面化表示方法，将三维体高斯投影为二维平面圆盘结构以增强几何约束能力，在提升表面重建精度的同时保持高效渲染性能，实现几何准确性与视觉质量的统一优化。</span></p></details>
在<u>光照正则化</u>方面，我们采用文献 _[32]_ 中提出的自然光照正则化形式：

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/45861457/1769410329056-5cae35ad-79f6-48ea-930d-aedc935b6a7e.png)

其中，$ \mathbf{L} $表示预测得到的环境光照，而$ \bar{\mathbf{L}} $表示其三个颜色通道的均值。

<details class="lake-collapse"><summary id="u1c07899c"><em><span class="ne-text">[32]</span></em><em><span class="ne-text" style="color: #1DC0C9">*</span></em></summary><p id="uaef8f406" class="ne-p"><span class="ne-text">[32] NeRO: Neural Geometry and BRDF Reconstruction of Reflective Objects from Multiview Images（ACM TOG 2023）：由 Yuan Liu、Peng Wang、Cheng Lin、Xiaoxiao Long、Jiepeng Wang、Lingjie Liu、Taku Komura 与 Wenping Wang 发表在 ACM Transactions on Graphics（TOG 2023），提出一种面向高反射物体的神经重建框架，通过联合优化</span><strong><span class="ne-text">几何形状与物理一致的 BRDF 参数</span></strong><span class="ne-text">，在多视图监督下实现可重光照的高质量重建。该方法将神经隐式表示与物理渲染模型紧密结合，有效缓解了强视角相关反射带来的重建歧义问题，是反射物体神经重建领域的重要进展。</span></p></details>
在<u> PBR 参数正则化</u>方面，我们借鉴文献 _[18]_ 的做法，<u>基于</u>这样<u>一种假设：在颜色变化平滑的区域内，材质属性不会发生剧烈变化</u>。相应的正则化项定义为：

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/45861457/1769410349162-26be9dfa-9ac5-49c3-bc4d-6ce3b2b051fc.png)

其中，$ \mathbf{X} $表示按照[式 (3)](#u5ef9d208) 渲染得到的 PBR 参数图，$ \mathbf{C}_{gt} $为真实颜色图像。

<details class="lake-collapse"><summary id="uf628dd98"><em><span class="ne-text">[18]</span></em></summary><p id="ub1f578b4" class="ne-p"><span class="ne-text">[18] Relightable 3D Gaussian: Real-Time Point Cloud Relighting with BRDF Decomposition and Ray Tracing（arXiv 2023）：由 Jian Gao、Chun Gu、Youtian Lin、Hao Zhu、Xun Cao、Li Zhang 与 Yao Yao 发布于 arXiv，提出一种可重光照的 3D Gaussian 表示方法，在 3D Gaussian Splatting 框架中引入 </span><strong><span class="ne-text">BRDF 分解与光线追踪机制</span></strong><span class="ne-text">，实现点云级别的物理一致重光照。该方法通过将外观表示从视角相关颜色扩展为物理材质参数，使 3DGS 同时具备实时渲染速度与可编辑光照能力，为显式高斯表示与 PBR 融合提供了重要方向。</span></p></details>
<u>综上，最终的总体损失函数</u>由各项损失加权组合而成，其表达式<u>为</u>：

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/45861457/1769410366603-a97ad923-4fba-40f7-92e7-89225ca6b6a3.png)

## Experiments
### Datasets（数据集）
<u>我们在一个合成数据集和一个真实世界数据集上进行了实验评估，分别是来自 NeRO </u>_<u>[32]</u>_<u> 的 </u><u><font style="background-color:#F8B881;">Glossy Blender 数据集</font></u><u>以及 </u><u><font style="background-color:#F8B881;">StanfordORB</font></u><u> </u>_<u>[27]</u>_<u> </u><u><font style="background-color:#F8B881;">数据集</font></u>。<u>Glossy Blender 数据集包含 8 个具有强镜面反射材质的物体，主要用于评估方法在重建质量和重光照效果方面的性能</u>。<u>StanfordORB 数据集包含 14 个具有多种材质属性的物体，每个物体均在 3 种不同的光照条件下进行采集。对于每个物体，我们随机选择其中一种光照条件用于实验</u>。

> Glossy Blender 数据集：[https://github.com/liuyuan-pal/NeRO?tab=readme-ov-file](https://github.com/liuyuan-pal/NeRO?tab=readme-ov-file)
>
> StanfordORB 数据集：[https://github.com/StanfordORB/Stanford-ORB](https://github.com/StanfordORB/Stanford-ORB)
>

<details class="lake-collapse"><summary id="ub5f9668a"><em><span class="ne-text">[32, 27]</span></em></summary><p id="u78bccfa0" class="ne-p"><span class="ne-text">[32] NeRO: Neural Geometry and BRDF Reconstruction of Reflective Objects from Multiview Images（ACM TOG 2023）：由 Yuan Liu、Peng Wang、Cheng Lin、Xiaoxiao Long、Jiepeng Wang、Lingjie Liu、Taku Komura 与 Wenping Wang 发表在 ACM Transactions on Graphics（TOG 2023），提出一种面向强反射物体的神经重建方法，在多视图监督下联合优化</span><strong><span class="ne-text">隐式几何表示与物理一致的 BRDF 参数</span></strong><span class="ne-text">，实现可重光照的高质量三维重建，有效缓解了视角相关反射带来的几何与材质耦合歧义问题。</span></p><p id="u72cffa27" class="ne-p"><span class="ne-text">[27] Stanford-ORB: A Real-World 3D Object Inverse Rendering Benchmark（NeurIPS 2024）：由 Zhengfei Kuang、Yunzhi Zhang、Hong-Xing Yu、Samir Agarwala、Elliott Wu、Jiajun Wu 等发表于 NeurIPS 2024，构建了一个面向真实世界物体的三维逆渲染基准数据集，提供高质量多视图图像及对应几何、材质与光照标注，用于系统评估几何恢复、BRDF 重建与可重光照能力，推动了逆渲染任务从合成数据向真实复杂场景过渡。</span></p></details>
<u>在数据预处理方面，我们使用</u>_<u>低动态范围（Low Dynamic Range，LDR）图像</u>_<u>作为输入，并将其下采样至</u>$ 1024 \times 1024 $<u>分辨率用于训练</u>。

<details class="lake-collapse"><summary id="u6fb54442"><em><span class="ne-text">低动态范围（Low Dynamic Range，LDR）图像 </span></em><span class="ne-text">：Glossy Blender 数据集 和 StanfordORB 数据集 提供 LDR 和 HDR 两种图像，作者选用 LDR 图像进行训练。</span></summary><p id="ua756c4a2" class="ne-p"><img src="https://cdn.nlark.com/yuque/0/2026/png/45861457/1771821981960-3c879d81-c392-458d-be35-52ddee9a572d.png" width="819.6302258265447" title="" crop="0,0,1,1" id="u81581093" class="ne-image"></p></details>
### Implementation Details（实现细节）
<u>在几何先验获取方面，我们采用由文献 </u>_<u>[34]</u>_<u> 微调后的 Marigold </u>_<u>[25]</u>_<u> 进行法线估计，并使用 Depth Pro</u>_<u> [3]</u>_<u> 进行深度估计</u>，该方法在目标级别预测任务上表现出色。<u>为保证训练过程的稳定性，我们将整体训练划分为两个阶段</u>。

<details class="lake-collapse"><summary id="ub604f1f8"><em><span class="ne-text">[34, 25, 3]</span></em><span class="ne-text"></span></summary><p id="u7c06391d" class="ne-p"><span class="ne-text">[34] Fine-tuning Image-conditional Diffusion Models Is Easier Than You Think（WACV 2025）：针对条件扩散模型微调过程复杂、计算开销大的问题，提出一种简单高效的微调策略，使图像条件扩散模型在较小计算资源下即可实现稳定收敛，并保持较高的生成质量与泛化能力。</span></p><p id="u801c2670" class="ne-p"><span class="ne-text">[25] Repurposing Diffusion-based Image Generators for Monocular Depth Estimation（CVPR 2024）：提出将预训练的扩散图像生成模型重新用于单目深度估计任务，通过利用扩散模型内部蕴含的语义与结构先验信息，实现高质量深度预测，在小样本条件下仍能保持较强的深度恢复能力。</span></p><p id="u3980cce8" class="ne-p"><span class="ne-text">[3] Depth Pro: Sharp Monocular Metric Depth in Less Than a Second（arXiv 2024）：提出一种高效的单目度量深度预测模型，采用高分辨率特征建模与优化的推理结构，在保证深度边界锐利度的同时实现接近实时的深度估计速度，显著提升了复杂场景下单目深度恢复的精度与实用性。</span></p></details>
<u>在第一阶段，我们在原始 2DGS 模型的基础上，引入额外的几何监督损失</u>$ \mathcal{L}_n $<u>和</u>$ \mathcal{L}_d $<u>进行训练</u>，其中损失权重设置为$ \lambda_n = 0.5 $、$ \lambda_d = 0.05 $，训练迭代次数为 30,000 次。除上述改动外，其余训练设置均与 2DGS 保持一致。

<u>在第二阶段，我们启用基于物理的渲染（PBR）管线，并对几何结构、PBR 参数以及环境光照进行联合优化</u>。各项损失的权重设置为$ \lambda_{\text{light}} = 0.002 $、$ \lambda_a = 0.05 $、$ \lambda_m = 0.05 $以及$ \lambda_r = 0.01 $。第二阶段额外进行 10,000 次迭代以完成优化。

<u>我们的方法基于 PyTorch 实现，所有实验均在单张 NVIDIA RTX 4090 GPU 上完成</u>。

### Baselines and metrics（对比方法与评估指标）
<u>在对比方法的选择上，我们选取了反射物体重建与重光照领域中具有代表性的最新方法作为基线进行比较</u>，具体包括以下几类：

**（1）基于 SDF 的方法**：  
NeRO _[32]_：一种基于 NeuS _[44]_ 并引入基于物理渲染（PBR）管线的方法；  
TensoSDF _[28]_：在 NeRO 的基础上，采用张量化表示并结合粗糙度感知的训练目标，以提升重建效果。

<details class="lake-collapse"><summary id="ue82909b5"><em><span class="ne-text">[32, 44, 28]</span></em></summary><p id="u4ad06d32" class="ne-p"><span class="ne-text">[32] NeRO: Neural Geometry and BRDF Reconstruction of Reflective Objects from Multiview Images（ACM TOG 2023）：由 Yuan Liu、Peng Wang、Cheng Lin、Xiaoxiao Long、Jiepeng Wang、Lingjie Liu、Taku Komura 与 Wenping Wang 发表在 ACM Transactions on Graphics（TOG 2023），提出一种面向强反射物体的神经重建方法，在多视图监督下联合优化</span><strong><span class="ne-text">隐式几何表示与物理一致的 BRDF 参数</span></strong><span class="ne-text">，实现可重光照的高质量三维重建，有效缓解了视角相关反射带来的几何与材质耦合歧义问题。</span></p><p id="u73470be4" class="ne-p"><span class="ne-text">[44] NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction（NeurIPS 2021）：由 Peng Wang、Lingjie Liu、Yuan Liu、Christian Theobalt、Taku Komura 与 Wenping Wang 发表在 NeurIPS 2021，提出一种基于体渲染的神经隐式曲面学习方法，通过引入符号距离函数（SDF）与体渲染概率密度的耦合优化策略，实现多视图场景下高质量连续曲面重建，解决传统隐式表示难以直接提取精确三角网格的问题，成为神经表面重建领域的代表性方法之一。</span></p><p id="u6983df4e" class="ne-p"><span class="ne-text">[28] </span><strong><span class="ne-text">TensoSDF: Roughness-aware Tensorial Representation for Robust Geometry and Material Reconstruction</span></strong><span class="ne-text">（ACM Transactions on Graphics, 2024）：提出粗糙度感知的张量化 SDF 表示方法，通过构建几何与材质耦合的高维参数张量实现复杂反射表面的稳定建模，在强光照变化与高反射场景中提升重建的鲁棒性与几何–材质一致性。</span></p></details>
**（2）基于 Gaussian 的方法**：  
GShader _[22]_：一种基于 3DGS 并融合 PBR 管线的方法；  
GS-IR _[31]_：通过引入遮挡建模与光照烘焙来提升渲染质量的方法；  
R3DG _[18]_：进一步通过基于点的光线追踪策略提升重光照质量的方法。

<details class="lake-collapse"><summary id="u98e321fd"><em><span class="ne-text">[22, 31, 18]</span></em></summary><p id="u922a38b8" class="ne-p"><span class="ne-text">[22] GaussianShader: 3D Gaussian Splatting with Shading Functions for Reflective Surfaces（CVPR 2024）：由 Yingwenqi Jiang、Jiadong Tu、Yuan Liu、Xifeng Gao、Xiaoxiao Long、Wenping Wang 与 Yuexin Ma 发表在 CVPR 2024，针对高反射表面场景，在 3D Gaussian Splatting 中引入</span><strong><span class="ne-text">可学习的着色函数（Shading Function）建模视角相关外观变化</span></strong><span class="ne-text">，在保持高渲染效率的同时显著提升复杂反射材质的重建质量，是高斯表示逆渲染方向的重要进展。</span></p><p id="ufcf34180" class="ne-p"><span class="ne-text">[31] GS-IR: 3D Gaussian Splatting for Inverse Rendering（CVPR 2024）：由 Zhihao Liang、Qi Zhang、Ying Feng、Ying Shan 与 Kui Jia 发表在 CVPR 2024，提出一种面向逆渲染任务的高斯表示方法，将场景分解为</span><strong><span class="ne-text">几何结构、材质属性与光照环境</span></strong><span class="ne-text">三个可优化模块，通过物理约束优化实现可重光照的三维重建，推动显式点状表示在 inverse rendering 中的应用。</span></p><p id="udd90cba1" class="ne-p"><span class="ne-text">[18] Relightable 3D Gaussian: Real-Time Point Cloud Relighting with BRDF Decomposition and Ray Tracing（arXiv 2023）：由 Jian Gao、Chun Gu、Youtian Lin、Hao Zhu、Xun Cao、Li Zhang 与 Yao Yao 发布于 arXiv，在 3D Gaussian Splatting 框架中引入 </span><strong><span class="ne-text">BRDF 分解与光线追踪重光照机制</span></strong><span class="ne-text">，实现点云级物理一致重光照渲染，使高斯表示不仅具备实时性优势，还具备材质可编辑能力。</span></p></details>
<u>在评估指标方面，我们采用 Chamfer Distance 来衡量三维重建的几何质量，并使用 PSNR 与 SSIM 指标对重光照效果进行定量评估</u>。

### Comparison on Reconstruction Quality（重建质量对比）
<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/45861457/1769411680613-fad2e7b2-912a-4c41-b952-e4c361c4f58f.png)

**表 1**. Blender Glossy 数据集上 3D 重建结果的 Chamfer-L1 ↓ 距离。

每个单元格按<font style="background-color:#F297CC;">最佳（best）</font>、<font style="background-color:#F8B881;">次优（second）</font>和<font style="background-color:#FCE75A;">第三（third）</font>进行标色。我们的方法在基于 GS 的方法中取得了最高的重建效果，并且在基于 SDF 的方法中性能位列第二，同时计算成本低一个数量级。

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/45861457/1769411409853-05d237ff-0c41-45fb-a7d1-7b710fa946ec.png)

**图 4**. Glossy Blender 数据集上的重建结果对比。

与其他基于 Gaussian 的方法相比，我们的方法取得了显著更优的重建效果，并且在整体性能上可与基于 SDF 的方法相媲美。同时，在图中标出的区域可以看到，我们的方法在细节刻画方面相较于 SDF 方法仍具有一定优势。

<u>我们在 Glossy Blender 数据集 </u>_<u>[32]</u>_<u> 和 StanfordORB 数据集 </u>_<u>[27]</u>_<u> 上对方法的重建质量进行了评估。在 Glossy Blender 数据集上，如</u>_<u>表 1</u>_<u> 所示，我们的方法在所有基于 Gaussian 的方法中取得了最佳的重建质量。与基于 SDF 的方法相比，我们的结果与 NeRO </u>_<u>[32]</u>_<u> 相当，并优于 TensoSDF </u>_<u>[28]</u>_。<u>定性对比结果如</u>_<u>图 4</u>_<u> 所示。与其他基于 Gaussian 的方法相比，我们的方法在重建质量上有显著提升，而其他方法往往存在较明显的噪声和表面起伏</u>。

<details class="lake-collapse"><summary id="u320bcf8a"><em><span class="ne-text">[32, 27, 28]</span></em></summary><p id="ueeffec23" class="ne-p"><span class="ne-text">[32] NeRO: Neural Geometry and BRDF Reconstruction of Reflective Objects from Multiview Images（ACM TOG 2023）：由 Yuan Liu、Peng Wang、Cheng Lin、Xiaoxiao Long、Jiepeng Wang、Lingjie Liu、Taku Komura 与 Wenping Wang 发表在 ACM Transactions on Graphics（TOG 2023），提出一种面向强反射物体的神经重建方法，在多视图监督下联合优化</span><strong><span class="ne-text">隐式几何表示与物理一致的 BRDF 参数</span></strong><span class="ne-text">，实现可重光照的高质量三维重建，有效缓解了视角相关反射带来的几何与材质耦合歧义问题。</span></p><p id="ud3c8f650" class="ne-p"><span class="ne-text">[27] Stanford-ORB: A Real-World 3D Object Inverse Rendering Benchmark（NeurIPS 2024）：由 Zhengfei Kuang、Yunzhi Zhang、Hong-Xing Yu、Samir Agarwala、Elliott Wu、Jiajun Wu 等发表于 NeurIPS 2024，构建了一个面向真实世界物体的三维逆渲染基准数据集，提供高质量多视图图像及对应几何、材质与光照标注，用于系统评估几何恢复、BRDF 重建与可重光照能力，推动了逆渲染任务从合成数据向真实复杂场景过渡。</span></p><p id="ue19ca8aa" class="ne-p"><span class="ne-text">[28] </span><strong><span class="ne-text">TensoSDF: Roughness-aware Tensorial Representation for Robust Geometry and Material Reconstruction</span></strong><span class="ne-text">（ACM Transactions on Graphics, 2024）：提出粗糙度感知的张量化 SDF 表示方法，通过构建几何与材质耦合的高维参数张量实现复杂反射表面的稳定建模，在强光照变化与高反射场景中提升重建的鲁棒性与几何–材质一致性。</span></p></details>
<u>基于 SDF 的方法由于 MLP 本身所带来的内在平滑先验，通常能够生成比我们方法更加平滑的表面。然而，由于 Gaussian 表示并不具备类似的约束，我们的结果在整体平滑性上略逊于 SDF 方法</u>。尽管如此，我们的方法能够更好地捕捉表面细节，例如马鬃区域以及桌铃的按压杆部分。<u>此外，我们还观察到，由于 SDF 的</u>_<u>封闭体（watertight）约束 </u>_<u>以及过度平滑效应，基于 SDF 的方法在未观测区域（如铃铛底部）往往会产生向外鼓起的伪表面，而我们的方法在这些区域能够得到更加干净、合理的重建结果</u>。

<details class="lake-collapse"><summary id="u616d2103"><em><span class="ne-text">封闭体（watertight）约束 </span></em><span class="ne-text">：指在重建三维物体时，模型假设目标物体的表面是一个几何上完全闭合、没有孔洞的实体。</span></summary><p id="u3347fde2" class="ne-p"><img src="https://cdn.nlark.com/yuque/0/2026/png/45861457/1771823154353-7b3cfd2a-cefa-4f42-b86f-650411ba57bd.png" width="819.6302258265447" title="" crop="0,0,1,1" id="u46c1950c" class="ne-image"></p></details>
<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/45861457/1769411748807-250de8d5-8567-4886-891f-3f9369de2cc3.png)

**表 2**. Stanford-ORB 数据集上 3D 重建结果的 Chamfer-L1 ↓ 距离，用以评估重建质量。

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/45861457/1769411476496-46cff4d2-5766-4e9a-8e77-bcdbcf717679.png)

**图 5**. StanfordORB 数据集上的结果对比。我们展示了重建得到的网格、渲染后的 PBR 参数以及法线图。其中，GShader 和 R3DG 方法不包含金属度（Metallic）结果。

<u>在 StanfordORB 数据集上的结果如</u>_<u>表 2</u>_<u> 所示，我们的方法在整个数据集上取得了最佳的重建质量，并在大多数情况下优于其他对比方法</u>。<u>从</u>_<u>图 5</u>_<u> 的对比结果可以看出，我们的方法在 PBR 参数分解方面表现更加合理，同时噪声水平更低</u>。

### Relighting Comparison（重光照效果对比）
<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/45861457/1769411800965-bb57109a-501d-414a-843d-87b8eca30789.png)

**表 3**. Blender Glossy 数据集上重光照结果的评估指标，包括 PSNR ↑ 和 SSIM ↓，报告了三种新环境光下的平均值。

对比结果表明，我们的方法在渲染质量上达到最高水平，同时帧率具有竞争力。

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/45861457/1769411532485-f76f8fd1-8ad2-462d-8af4-3f054ed020e3.png)

**图 6**. Glossy Blender 数据集上的重光照（Relighting）结果。

<u>我们进一步在 Glossy Blender 数据集上评估了方法的重光照质量，其定量结果如</u>_<u>表 3</u>_<u> 所示</u>。<u>我们的方式在所有基于 Gaussian 的方法中取得了最高的重光照性能</u>。_<u>图 6</u>_<u> 中的定性对比结果进一步表明，得益于我们方法所估计的 PBR 参数与几何信息，能够生成更具真实感和一致性的重光照效果，而其他方法则难以产生合理的重光照结果</u>。

### Ablation Study（消融实验）
<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/45861457/1769411855562-f6c62a75-65a3-47db-aa2d-7079dc5c03f5.png)

**表 4**. Glossy Blender 数据集上的消融实验结果。结果表明，来自基础模型的几何监督对重建性能的提升最为显著，而延迟着色（Deferred Shading）技术则有助于提高渲染质量。

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/45861457/1769411579448-dff27e69-b37a-4ba0-a271-ae81ac0fef01.png)

**图 7**. 延迟着色（Deferred Shading）与前向着色（Forward Shading）在环境光照估计质量上的对比。

<u>我们通过在 Glossy Bender 数据集上开展消融实验来验证所提出方法中各个组成模块的有效性，结果如</u>_<u>表 4</u>_<u> 所示</u>。<u>以 2DGS 作为基线方法，首先引入来自基础模型的几何监督，可以显著提升重建质量</u>。然而，由于模型灵活性受限，渲染质量有所下降。<u>随后，我们进一步引入基于物理的渲染（PBR）以刻画反射特性，在重建质量和渲染质量两方面均取得了提升</u>。<u>最后，我们加入延迟着色（deferred shading）模块，形成完整模型</u>。<u>正如第 </u>[<u>1.3.5 节</u>](#bOne5)<u>所分析的，延迟着色能够更准确地分离和建模各类着色分量，从而提升环境光照的估计精度</u>。_<u>图 7</u>_<u> 对估计得到的环境光照结果进行了可视化</u>。<u>更精确的环境光照估计进一步带来了更优的渲染质量</u>。

## Conclusion
本文提出了 **GS-2DGS**，一种专为反射物体设计的高斯基重建方法。通过引入基于物理的渲染（PBR）管线，模型能够显式刻画物体表面与环境光照之间的相互作用，从而有效模拟反射物体的真实外观特性。在此基础上，我们进一步融合了来自基础模型的几何监督，并引入延迟着色（deferred shading）机制，使得该方法在较低计算成本下即可实现高质量的几何重建与逼真的重光照效果。

尽管取得了上述进展并显著降低了计算开销，与当前最先进的基于 SDF 的方法相比，我们的方法在性能上仍存在一定差距。我们认为这一差距主要源于高斯表示在几何约束方面的先天不足。尽管已有部分并行工作尝试引入 SDF 作为额外的几何监督以弥补这一问题，但这类方法通常伴随着较高的计算成本。未来的研究可以进一步探索如何在兼顾重建精度的同时保持高效计算，从而实现速度与准确性的统一，这也为后续工作提供了一个具有吸引力的研究方向。



