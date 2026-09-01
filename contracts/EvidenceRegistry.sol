// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title EvidenceRegistry
/// @notice First-seen timestamping for face-derived evidence bundles.
/// @dev Design constraints, all deliberate:
///
///      * NOTHING BIOMETRIC IS STORED. `subjectCommitment` is
///        SHA-256(salt || quantized_embedding) with the salt held off-chain.
///        A public ledger is immutable and world-readable; publishing a face
///        template there could never be undone. The commitment still lets us
///        later prove a record concerns a given subject, by revealing the salt.
///
///      * ONLY THE DOMAIN, NOT THE FULL URL. The full URL lives in the evidence
///        bundle and is covered by `evidenceHash`, so it is provably fixed at
///        record time without being permanently broadcast.
///
///      * The chain notarises WHEN a claim was made. It does not, and cannot,
///        validate that the claim is true.
contract EvidenceRegistry {
    struct Record {
        uint64  timestamp;
        address recorder;
        bytes32 subjectCommitment;
        string  sourceDomain;
    }

    mapping(bytes32 => Record) private records;

    event EvidenceRecorded(
        bytes32 indexed evidenceHash,
        bytes32 indexed subjectCommitment,
        address indexed recorder,
        uint64  timestamp,
        string  sourceDomain
    );

    error AlreadyRecorded(bytes32 evidenceHash, uint64 firstSeen);

    /// @notice Record an evidence hash. Reverts if already present, which is
    ///         what gives the registry first-seen semantics rather than
    ///         last-writer-wins.
    function recordEvidence(
        bytes32 evidenceHash,
        bytes32 subjectCommitment,
        string calldata sourceDomain
    ) external {
        Record storage r = records[evidenceHash];
        if (r.timestamp != 0) revert AlreadyRecorded(evidenceHash, r.timestamp);
        records[evidenceHash] = Record(
            uint64(block.timestamp), msg.sender, subjectCommitment, sourceDomain
        );
        emit EvidenceRecorded(
            evidenceHash, subjectCommitment, msg.sender, uint64(block.timestamp), sourceDomain
        );
    }

    /// @notice Read a record. `exists` is false for an unregistered hash.
    function getEvidence(bytes32 evidenceHash)
        external
        view
        returns (
            bool exists,
            uint64 timestamp,
            address recorder,
            bytes32 subjectCommitment,
            string memory sourceDomain
        )
    {
        Record storage r = records[evidenceHash];
        return (r.timestamp != 0, r.timestamp, r.recorder, r.subjectCommitment, r.sourceDomain);
    }
}
