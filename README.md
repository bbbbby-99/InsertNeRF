# InsertNeRF: Instilling Generalizability into NeRF with HyperNet Modules

## Introduction
Generalizing Neural Radiance Fields (NeRF) to new scenes is a significant challenge that existing approaches struggle to address without extensive modifications to vanilla NeRF framework. We introduce **InsertNeRF**, a method for **INS**tilling g**E**ne**R**alizabili**T**y into **NeRF**. By utilizing multiple plug-and-play HyperNet modules, InsertNeRF dynamically tailors NeRF's weights to specific reference scenes, transforming multi-scale sampling-aware features into scene-specific representations. This novel design allows for more accurate and efficient representations of complex appearances and geometries. Experiments show that this method not only achieves superior generalization performance but also provides a flexible  pathway for integration with other NeRF-like systems, even in sparse input settings.

<div style="display: flex; justify-content: space-between;">
    <img src="https://github.com/bbbbby-99/InsertNeRF/blob/main/gif%26image/lego.gif" alt="Lego GIF" width="200" />
    <img src="https://github.com/bbbbby-99/InsertNeRF/blob/main/gif%26image/chair.gif" alt="Chair GIF" width="200" />
    <img src="https://github.com/bbbbby-99/InsertNeRF/blob/main/gif%26image/ficus.gif" alt="Ficus GIF" width="200" />
    <img src="https://github.com/bbbbby-99/InsertNeRF/blob/main/gif%26image/mic.gif" alt="Mic GIF" width="200" />
</div>
