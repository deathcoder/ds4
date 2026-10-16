<!--
{
  "availability" : [
    "iOS: 8.0.0 -",
    "iPadOS: 8.0.0 -",
    "macCatalyst: 13.1.0 -",
    "macOS: 10.11.0 -",
    "tvOS: -",
    "visionOS: 1.0.0 -"
  ],
  "documentType" : "symbol",
  "framework" : "Metal",
  "identifier" : "/documentation/Metal/MTLCommandBuffer/retainedReferences",
  "metadataVersion" : "0.1.0",
  "role" : "Instance Property",
  "symbol" : {
    "kind" : "Instance Property",
    "modules" : [
      "Metal"
    ],
    "preciseIdentifier" : "c:objc(pl)MTLCommandBuffer(py)retainedReferences"
  },
  "title" : "retainedReferences"
}
-->

# retainedReferences

A Boolean value that indicates whether the command buffer maintains strong references to the resources it uses.

```
var retainedReferences: Bool { get }
```

## Discussion

You can configure this property when you create a command buffer by setting [`retainedReferences`](/documentation/Metal/MTLCommandBufferDescriptor/retainedReferences) of an [`MTLCommandBufferDescriptor`](/documentation/Metal/MTLCommandBufferDescriptor) instance and calling the [`makeCommandBuffer(descriptor:)`](/documentation/Metal/MTLCommandQueue/makeCommandBuffer(descriptor:)) method. The [`makeCommandBuffer()`](/documentation/Metal/MTLCommandQueue/makeCommandBuffer()) method sets this property to <doc://com.apple.documentation/documentation/Swift/true>, and [`makeCommandBufferWithUnretainedReferences()`](/documentation/Metal/MTLCommandQueue/makeCommandBufferWithUnretainedReferences()) sets it to <doc://com.apple.documentation/documentation/Swift/false>.

If <doc://com.apple.documentation/documentation/Swift/false>, your app is responsible for maintaining strong references to all the resources the command buffer relies on until it completes.

> Important:
> Releasing a resource before a command buffer’s commands complete may cause a runtime error or erratic behavior.

---

Copyright &copy; 2026 Apple Inc. All rights reserved. | [Terms of Use](https://www.apple.com/legal/internet-services/terms/site.html) | [Privacy Policy](https://www.apple.com/privacy/privacy-policy)