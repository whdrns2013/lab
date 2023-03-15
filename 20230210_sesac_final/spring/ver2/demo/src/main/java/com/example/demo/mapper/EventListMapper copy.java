package com.example.demo.mapper;

import java.util.ArrayList;

import org.apache.ibatis.annotations.Mapper;

import com.example.demo.model.EventList;

@Mapper
public interface EventListMapper {

    public ArrayList<EventList> selectAll();
    
}