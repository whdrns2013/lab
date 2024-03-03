import 'package:flutter/material.dart';

class CustomSearchBar extends StatelessWidget {
  double barWidth; // 600
  double barHeight; // 50

  CustomSearchBar({
    super.key,
    required this.barWidth,
    required this.barHeight,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 20, 20, 20),
      child: Container(
        width: barWidth,
        height: barHeight,
        decoration: BoxDecoration(
          border: Border.all(width: 1.5),
          borderRadius: BorderRadius.circular(25),
          color: Colors.white,
        ),
        child: Row(
          children: [
            SizedBox(
              width: barWidth / 12,
            ),
            Container(
              width: barWidth * 10 / 12,
              child: TextField(
                decoration: InputDecoration(
                  border: InputBorder.none,
                ),
              ),
            ),
            IconButton(
              onPressed: send_mesasge,
              icon: Icon(
                Icons.search,
                size: barWidth / 24,
              ),
              iconSize: barWidth / 24,
            ),
          ],
        ),
      ),
    );
  }
}

void send_mesasge() {}
