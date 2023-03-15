package com.example.demo.mapper;

import java.util.ArrayList;

import org.apache.ibatis.annotations.Mapper;

import com.example.demo.model.EventList;

@Mapper
public interface DetailMapper {

    public ArrayList<EventList> select(event_name);
    
}
